"""Fail-closed parsing for bounded OpenMC DAGMC geometry-debug runs."""

from __future__ import annotations

import re
from typing import Any


REQUIRED_OPENMC_VERSION = "0.16.0"
REQUIRED_OPENMC_COMMIT = "617d35a5063c57796b43428bc401e627d2011046"


OVERLAP_PATTERN = re.compile(
    r"^\s*ERROR:\s*Overlapping cells detected:\s*(\d+)\s*,\s*(\d+)"
    r"\s+on universe\s+(\d+)",
    re.IGNORECASE | re.MULTILINE,
)

LOST_OR_NAVIGATION_PATTERNS = (
    (
        "cell_not_found",
        re.compile(
            r"Could not find the cell containing particle", re.IGNORECASE
        ),
    ),
    (
        "surface_crossing_not_located",
        re.compile(
            r"After particle .* crossed surface .* it could not be located "
            r"in any cell and it did not leak",
            re.IGNORECASE,
        ),
    ),
    (
        "collision_near_surface_not_located",
        re.compile(
            r"Could not find particle after a collision near a surface",
            re.IGNORECASE,
        ),
    ),
    (
        "lattice_boundary_not_located",
        re.compile(
            r"Particle .* could not be located after crossing a boundary "
            r"of lattice",
            re.IGNORECASE,
        ),
    ),
    (
        "negative_lattice_distance",
        re.compile(
            r"Particle .* had a negative distance to a lattice boundary",
            re.IGNORECASE,
        ),
    ),
    (
        "reflection_not_located",
        re.compile(r"Couldn't find particle after reflecting", re.IGNORECASE),
    ),
    (
        "periodic_not_located",
        re.compile(
            r"Couldn't find particle after hitting periodic", re.IGNORECASE
        ),
    ),
    (
        "reflect_failure",
        re.compile(r"Cannot reflect particle", re.IGNORECASE),
    ),
    (
        "transfer_failure",
        re.compile(r"Cannot transfer particle", re.IGNORECASE),
    ),
    (
        "dagmc_no_intersection",
        re.compile(
            r"No intersection found with DAGMC cell .* filled with material",
            re.IGNORECASE,
        ),
    ),
    (
        "maximum_particle_events",
        re.compile(
            r"Particle\s+\d+\s+underwent maximum number of events\.",
            re.IGNORECASE,
        ),
    ),
    (
        "lattice_without_outer_definition",
        re.compile(
            r"Particle\s+\d+\s+left lattice\s+\d+,\s*but it has no outer "
            r"definition\.",
            re.IGNORECASE,
        ),
    ),
)

MAXIMUM_LOST_PATTERN = re.compile(
    r"ERROR:\s*Maximum number of lost particles has been reached",
    re.IGNORECASE,
)
ERROR_LINE_PATTERN = re.compile(r"^\s*ERROR:\s*.*$", re.MULTILINE)


def parse_openmc_geometry_debug_log(
    text: str,
    *,
    exit_code: int,
    expected_threads: int | None = None,
    required_cell_ids: list[int] | tuple[int, ...] | None = None,
    expected_openmc_version: str = REQUIRED_OPENMC_VERSION,
    expected_openmc_commit: str = REQUIRED_OPENMC_COMMIT,
) -> dict[str, Any]:
    """Parse one OpenMC 0.16 geometry-debug log without optimistic defaults."""
    overlaps = [
        {
            "cell_ids": [int(match.group(1)), int(match.group(2))],
            "universe_id": int(match.group(3)),
        }
        for match in OVERLAP_PATTERN.finditer(text)
    ]
    unique_overlaps = sorted(
        {
            (
                row["cell_ids"][0],
                row["cell_ids"][1],
                row["universe_id"],
            )
            for row in overlaps
        }
    )
    diagnostics = {
        name: len(pattern.findall(text))
        for name, pattern in LOST_OR_NAVIGATION_PATTERNS
    }
    maximum_lost = len(MAXIMUM_LOST_PATTERN.findall(text))
    known_error_lines = {
        match.group(0).strip() for match in OVERLAP_PATTERN.finditer(text)
    }
    known_error_lines.update(
        line.strip()
        for line in ERROR_LINE_PATTERN.findall(text)
        if MAXIMUM_LOST_PATTERN.search(line)
    )
    unexpected_error_lines = [
        line.strip()
        for line in ERROR_LINE_PATTERN.findall(text)
        if line.strip() not in known_error_lines
    ]
    markers = {
        "simulating_batch_1": bool(
            re.search(r"Simulating batch\s+1\b", text, re.IGNORECASE)
        ),
        "simulating_batch_2": bool(
            re.search(r"Simulating batch\s+2\b", text, re.IGNORECASE)
        ),
        "results": bool(re.search(r"\bRESULTS\b", text)),
        "cell_overlap_check_summary": bool(
            re.search(r"CELL OVERLAP CHECK SUMMARY", text, re.IGNORECASE)
        ),
    }
    coverage_sections = re.findall(
        r"CELL OVERLAP CHECK SUMMARY[^\r\n]*\r?\n+"
        r"\s*Cell ID\s+No\. Overlap Checks\s*\n"
        r"(?P<rows>(?:\s*\d+\s+\d+\s*\r?\n)+)",
        text,
        re.IGNORECASE,
    )
    coverage_rows = []
    if len(coverage_sections) == 1:
        coverage_rows = [
            {"cell_id": int(cell_id), "overlap_checks": int(count)}
            for cell_id, count in re.findall(
                r"^\s*(\d+)\s+(\d+)\s*$",
                coverage_sections[0],
                re.MULTILINE,
            )
        ]
    coverage_ids = [row["cell_id"] for row in coverage_rows]
    required_cells = sorted(
        {int(value) for value in (required_cell_ids or ())}
    )
    coverage_by_id = {
        row["cell_id"]: row["overlap_checks"] for row in coverage_rows
    }
    duplicate_coverage_ids = sorted(
        {value for value in coverage_ids if coverage_ids.count(value) > 1}
    )
    missing_required_cells = sorted(set(required_cells) - set(coverage_ids))
    zero_required_cells = sorted(
        cell_id
        for cell_id in required_cells
        if coverage_by_id.get(cell_id, 0) <= 0
    )
    coverage_pass = (
        len(coverage_sections) == 1
        and bool(coverage_rows)
        and not duplicate_coverage_ids
        and not missing_required_cells
        and not zero_required_cells
    )
    version_matches = re.findall(
        r"^\s*Version\s*\|\s*([^\s]+)\s*$", text, re.MULTILINE
    )
    commit_matches = re.findall(
        r"^\s*Commit Hash\s*\|\s*([0-9a-fA-F]{40})\s*$",
        text,
        re.MULTILINE,
    )
    actual_version = version_matches[0] if len(version_matches) == 1 else None
    actual_commit = (
        commit_matches[0].lower() if len(commit_matches) == 1 else None
    )
    runtime_identity_pass = (
        len(version_matches) == 1
        and len(commit_matches) == 1
        and actual_version == expected_openmc_version
        and actual_commit == expected_openmc_commit.lower()
    )
    thread_matches = re.findall(
        r"OpenMP Threads\s*\|\s*(\d+)", text, re.IGNORECASE
    )
    actual_threads = (
        int(thread_matches[0]) if len(thread_matches) == 1 else None
    )
    thread_count_pass = expected_threads is None or (
        len(thread_matches) == 1 and actual_threads == int(expected_threads)
    )
    navigation_count = sum(diagnostics.values())
    passed = (
        int(exit_code) == 0
        and not overlaps
        and navigation_count == 0
        and maximum_lost == 0
        and not unexpected_error_lines
        and all(markers.values())
        and thread_count_pass
        and coverage_pass
        and runtime_identity_pass
    )
    return {
        "schema": "parastell.openmc_geometry_debug_log/v1.1.0",
        "exit_code": int(exit_code),
        "overlap_message_count": len(overlaps),
        "unique_overlap_tuples": [list(row) for row in unique_overlaps],
        "lost_or_navigation_diagnostics": diagnostics,
        "lost_or_navigation_count": navigation_count,
        "maximum_lost_terminal_count": maximum_lost,
        "unexpected_error_lines": unexpected_error_lines,
        "completion_markers": markers,
        "overlap_check_coverage": {
            "summary_match_count": len(coverage_sections),
            "rows": coverage_rows,
            "required_cell_ids": required_cells,
            "duplicate_cell_ids": duplicate_coverage_ids,
            "missing_required_cell_ids": missing_required_cells,
            "zero_check_required_cell_ids": zero_required_cells,
            "minimum_required_cell_checks": (
                min(coverage_by_id[cell_id] for cell_id in required_cells)
                if required_cells and not missing_required_cells
                else None
            ),
            "pass": coverage_pass,
        },
        "openmp_thread_match_count": len(thread_matches),
        "actual_openmp_threads": actual_threads,
        "expected_openmp_threads": (
            int(expected_threads) if expected_threads is not None else None
        ),
        "openmp_thread_count_pass": thread_count_pass,
        "runtime_identity": {
            "version_match_count": len(version_matches),
            "commit_match_count": len(commit_matches),
            "actual_version": actual_version,
            "actual_commit": actual_commit,
            "expected_version": expected_openmc_version,
            "expected_commit": expected_openmc_commit,
            "pass": runtime_identity_pass,
        },
        "pass": passed,
    }


def categorical_geometry_debug_signature(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Remove stochastic counts while retaining every diagnostic category."""
    coverage = report.get("overlap_check_coverage", {})
    diagnostics = report.get("lost_or_navigation_diagnostics", {})
    return {
        "schema": report.get("schema"),
        "exit_code": report.get("exit_code"),
        "unique_overlap_tuples": report.get("unique_overlap_tuples"),
        "diagnostic_categories_present": sorted(
            name for name, count in diagnostics.items() if int(count) > 0
        ),
        "maximum_lost_present": bool(
            report.get("maximum_lost_terminal_count", 0)
        ),
        "unexpected_error_lines": report.get("unexpected_error_lines"),
        "completion_markers": report.get("completion_markers"),
        "coverage_summary_match_count": coverage.get("summary_match_count"),
        "coverage_required_cell_ids": coverage.get("required_cell_ids"),
        "coverage_duplicate_cell_ids": coverage.get("duplicate_cell_ids"),
        "coverage_missing_required_cell_ids": coverage.get(
            "missing_required_cell_ids"
        ),
        "coverage_zero_check_required_cell_ids": coverage.get(
            "zero_check_required_cell_ids"
        ),
        "coverage_pass": coverage.get("pass"),
        "runtime_identity": report.get("runtime_identity"),
        "pass": report.get("pass"),
    }


def _rederive_log_gate(log: dict[str, Any], *, expected_threads: int) -> bool:
    diagnostics = log.get("lost_or_navigation_diagnostics", {})
    coverage = log.get("overlap_check_coverage", {})
    identity = log.get("runtime_identity", {})
    markers = log.get("completion_markers", {})
    expected_diagnostics = {name for name, _ in LOST_OR_NAVIGATION_PATTERNS}
    expected_markers = {
        "simulating_batch_1",
        "simulating_batch_2",
        "results",
        "cell_overlap_check_summary",
    }
    rows = coverage.get("rows")
    required = coverage.get("required_cell_ids")
    coverage_rederived = False
    if isinstance(rows, list) and isinstance(required, list):
        try:
            cell_ids = [int(row["cell_id"]) for row in rows]
            counts = [int(row["overlap_checks"]) for row in rows]
            required_ids = [int(value) for value in required]
            duplicate_ids = sorted(
                {value for value in cell_ids if cell_ids.count(value) > 1}
            )
            missing_ids = sorted(set(required_ids) - set(cell_ids))
            by_id = dict(zip(cell_ids, counts))
            zero_ids = sorted(
                value for value in required_ids if by_id.get(value, 0) <= 0
            )
            minimum = (
                min(by_id[value] for value in required_ids)
                if required_ids and not missing_ids
                else None
            )
            coverage_rederived = bool(
                rows
                and required_ids == sorted(set(required_ids))
                and all(value >= 0 for value in counts)
                and coverage.get("duplicate_cell_ids") == duplicate_ids == []
                and coverage.get("missing_required_cell_ids")
                == missing_ids
                == []
                and coverage.get("zero_check_required_cell_ids")
                == zero_ids
                == []
                and coverage.get("minimum_required_cell_checks") == minimum
                and minimum is not None
                and minimum > 0
            )
        except (KeyError, TypeError, ValueError):
            coverage_rederived = False
    return bool(
        log.get("schema") == "parastell.openmc_geometry_debug_log/v1.1.0"
        and log.get("exit_code") == 0
        and log.get("overlap_message_count") == 0
        and log.get("unique_overlap_tuples") == []
        and isinstance(diagnostics, dict)
        and set(diagnostics) == expected_diagnostics
        and all(int(value) == 0 for value in diagnostics.values())
        and log.get("lost_or_navigation_count")
        == sum(int(value) for value in diagnostics.values())
        and log.get("maximum_lost_terminal_count") == 0
        and log.get("unexpected_error_lines") == []
        and isinstance(markers, dict)
        and set(markers) == expected_markers
        and all(value is True for value in markers.values())
        and coverage.get("summary_match_count") == 1
        and coverage_rederived
        and coverage.get("pass") is True
        and log.get("openmp_thread_match_count") == 1
        and log.get("actual_openmp_threads") == int(expected_threads)
        and log.get("expected_openmp_threads") == int(expected_threads)
        and log.get("openmp_thread_count_pass") is True
        and identity.get("actual_version") == REQUIRED_OPENMC_VERSION
        and identity.get("expected_version") == REQUIRED_OPENMC_VERSION
        and identity.get("actual_commit") == REQUIRED_OPENMC_COMMIT
        and identity.get("expected_commit") == REQUIRED_OPENMC_COMMIT
        and identity.get("pass") is True
        and log.get("pass") is True
    )


def aggregate_geometry_debug_replicas(
    replicas: list[dict[str, Any]],
    *,
    required_seeds: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """Require one bound passing replica for every distinct registered seed."""
    seeds = [int(row.get("seed", -1)) for row in replicas]
    required = sorted(int(value) for value in required_seeds)
    unique_seeds = sorted(set(seeds))
    binding_fields = (
        "raw_h5m_sha256",
        "source_mesh_sha256",
        "acceptance_criteria_sha256",
        "openmc_version",
        "openmc_commit",
        "threads",
        "source_domain_audit_sha256",
        "source_point_cm",
        "reference_source_mesh_sha256",
        "reference_source_fingerprint",
        "cross_sections_sha256",
        "nuclear_data_manifest_sha256",
        "model_xml_seed_normalized_fingerprint",
        "histories",
        "batches",
        "particles_per_batch",
    )
    consistent_bindings = {
        field: sorted({str(row.get(field)) for row in replicas})
        for field in binding_fields
    }
    bindings_pass = all(
        len(values) == 1 and values[0] not in {"None", ""}
        for values in consistent_bindings.values()
    )
    replica_contract_passes = []
    for row in replicas:
        log = row.get("log_qualification", {})
        state = row.get("statepoint_summary_qualification", {})
        expected_threads = int(row.get("threads", -1))
        replica_contract_passes.append(
            row.get("schema")
            == "parastell.openmc_geometry_debug_replica/v1.2.0"
            and row.get("timed_out") is False
            and row.get("exit_code") == 0
            and row.get("input_binding_pass") is True
            and row.get("transport_input_immutability_pass") is True
            and row.get("model_xml_immutability_pass") is True
            and row.get("openmc_version") == REQUIRED_OPENMC_VERSION
            and row.get("openmc_commit") == REQUIRED_OPENMC_COMMIT
            and _rederive_log_gate(log, expected_threads=expected_threads)
            and state.get("readable") is True
            and state.get("summary_geometry_present") is True
            and state.get("summary_contract_pass") is True
            and state.get("statepoint_batches") == row.get("batches") == 2
            and state.get("statepoint_particles")
            == row.get("particles_per_batch")
            == 2000
            and state.get("statepoint_run_mode") == "fixed source"
            and state.get("statepoint_seed") == row.get("seed")
            and state.get("statepoint_seed_matches_expected") is True
            and row.get("histories") == 4000
            and row.get("lost_particle_restart_files") == []
            and row.get("replica_gate_pass") is True
        )
    gate = (
        len(replicas) == len(required)
        and len(seeds) == len(unique_seeds)
        and unique_seeds == required
        and bindings_pass
        and all(replica_contract_passes)
    )
    return {
        "schema": "parastell.openmc_geometry_debug_aggregate/v1.0.0",
        "required_seeds": required,
        "observed_seeds": seeds,
        "unique_observed_seeds": unique_seeds,
        "replica_count": len(replicas),
        "bindings": consistent_bindings,
        "bindings_pass": bindings_pass,
        "replica_contract_passes": replica_contract_passes,
        "replicas": replicas,
        "openmc_geometry_debug_gate_pass": gate,
    }
