import pytest
from types import SimpleNamespace

from parastell.openmc_geometry_debug import aggregate_geometry_debug_replicas
from parastell.openmc_geometry_debug import (
    categorical_geometry_debug_signature,
)
from parastell.openmc_geometry_debug import parse_openmc_geometry_debug_log
from scripts.run_openmc_geometry_debug_replicas import (
    _summary_material_id_names,
)


CLEAN_LOG = """
 Version | 0.16.0
 Commit Hash | 617d35a5063c57796b43428bc401e627d2011046
 OpenMP Threads | 4
 Simulating batch 1
 Simulating batch 2
 ============================>     RESULTS     <============================
 CELL OVERLAP CHECK SUMMARY

 Cell ID      No. Overlap Checks
       1                20
"""


def test_clean_openmc_geometry_debug_log_passes():
    report = parse_openmc_geometry_debug_log(CLEAN_LOG, exit_code=0)

    assert report["pass"] is True
    assert report["overlap_message_count"] == 0
    assert report["lost_or_navigation_count"] == 0
    assert all(report["completion_markers"].values())


def test_real_openmc_banner_suffix_and_runtime_identity_are_required():
    decorated = CLEAN_LOG.replace(
        "CELL OVERLAP CHECK SUMMARY",
        "CELL OVERLAP CHECK SUMMARY     <===================",
    )
    passed = parse_openmc_geometry_debug_log(decorated, exit_code=0)
    wrong_commit = parse_openmc_geometry_debug_log(
        decorated.replace(
            "617d35a5063c57796b43428bc401e627d2011046", "0" * 40
        ),
        exit_code=0,
    )

    assert passed["pass"] is True
    assert passed["overlap_check_coverage"]["summary_match_count"] == 1
    assert wrong_commit["pass"] is False
    assert wrong_commit["runtime_identity"]["pass"] is False


def test_categorical_signature_ignores_stochastic_counts_and_thread_layout():
    four_thread = parse_openmc_geometry_debug_log(
        CLEAN_LOG, exit_code=0, expected_threads=4, required_cell_ids=[1]
    )
    one_thread = parse_openmc_geometry_debug_log(
        CLEAN_LOG.replace("OpenMP Threads | 4", "OpenMP Threads | 1").replace(
            "1                20", "1                99"
        ),
        exit_code=0,
        expected_threads=1,
        required_cell_ids=[1],
    )

    assert categorical_geometry_debug_signature(
        four_thread
    ) == categorical_geometry_debug_signature(one_thread)


def test_summary_material_inventory_uses_summary_not_dagmc_graph():
    summary = SimpleNamespace(
        materials={
            2: SimpleNamespace(name="first_wall"),
            1: SimpleNamespace(name="Vacuum"),
        }
    )
    assert _summary_material_id_names(summary) == [
        [1, "Vacuum"],
        [2, "first_wall"],
    ]


def test_openmc_log_binds_reported_thread_count():
    passed = parse_openmc_geometry_debug_log(
        CLEAN_LOG,
        exit_code=0,
        expected_threads=4,
    )
    failed = parse_openmc_geometry_debug_log(
        CLEAN_LOG.replace("OpenMP Threads | 4", "OpenMP Threads | 1"),
        exit_code=0,
        expected_threads=4,
    )

    assert passed["pass"] is True
    assert passed["actual_openmp_threads"] == 4
    assert failed["pass"] is False


def test_repeated_overlap_messages_are_counted_and_deduplicated():
    line = "ERROR: Overlapping cells detected: 2, 3 on universe 1\n"
    report = parse_openmc_geometry_debug_log(
        CLEAN_LOG + line + line, exit_code=0
    )

    assert report["pass"] is False
    assert report["overlap_message_count"] == 2
    assert report["unique_overlap_tuples"] == [[2, 3, 1]]


@pytest.mark.parametrize(
    "diagnostic",
    [
        "Could not find the cell containing particle 7",
        "After particle 7 crossed surface 3 it could not be located in any cell and it did not leak",
        "Could not find particle after a collision near a surface",
        "Particle 7 could not be located after crossing a boundary of lattice 2",
        "Particle 7 had a negative distance to a lattice boundary",
        "Couldn't find particle after reflecting",
        "Couldn't find particle after hitting periodic boundary",
        "Cannot reflect particle 7",
        "Cannot transfer particle 7",
        "No intersection found with DAGMC cell 3, filled with material 4",
        "Particle 7 underwent maximum number of events.",
        "Particle 7 left lattice 2, but it has no outer definition.",
    ],
)
def test_every_pinned_navigation_diagnostic_fails(diagnostic):
    report = parse_openmc_geometry_debug_log(
        CLEAN_LOG + diagnostic + "\n", exit_code=0
    )

    assert report["pass"] is False
    assert report["lost_or_navigation_count"] == 1


def test_generic_error_nonzero_exit_and_truncated_logs_fail():
    generic = parse_openmc_geometry_debug_log(
        CLEAN_LOG + "ERROR: unexpected fatal condition\n", exit_code=0
    )
    nonzero = parse_openmc_geometry_debug_log(CLEAN_LOG, exit_code=139)
    truncated = parse_openmc_geometry_debug_log(
        "Simulating batch 1\n", exit_code=0
    )

    assert generic["pass"] is False
    assert generic["unexpected_error_lines"]
    assert nonzero["pass"] is False
    assert truncated["pass"] is False


def test_missing_graveyard_warning_is_not_naively_counted_as_lost_particle():
    report = parse_openmc_geometry_debug_log(
        CLEAN_LOG
        + "WARNING: No boundary conditions were applied to any surfaces. "
        "You should verify that your geometry does not allow particles to "
        "leak out; otherwise lost particles may result.\n",
        exit_code=0,
    )

    assert report["pass"] is True
    assert report["lost_or_navigation_count"] == 0


def test_overlap_coverage_requires_rows_but_reports_zero_check_cells():
    passed = parse_openmc_geometry_debug_log(
        CLEAN_LOG, exit_code=0, required_cell_ids=[1]
    )
    missing = parse_openmc_geometry_debug_log(
        CLEAN_LOG, exit_code=0, required_cell_ids=[1, 2]
    )
    zero = parse_openmc_geometry_debug_log(
        CLEAN_LOG.replace("1                20", "1                 0"),
        exit_code=0,
        required_cell_ids=[1],
    )

    assert passed["pass"] is True
    assert missing["pass"] is False
    assert missing["overlap_check_coverage"]["missing_required_cell_ids"] == [
        2
    ]
    assert zero["pass"] is False
    assert zero["overlap_check_coverage"]["zero_check_required_cell_ids"] == [
        1
    ]


def _replica(seed):
    return {
        "schema": "parastell.openmc_geometry_debug_replica/v1.2.0",
        "seed": seed,
        "raw_h5m_sha256": "a" * 64,
        "source_mesh_sha256": "b" * 64,
        "acceptance_criteria_sha256": "c" * 64,
        "openmc_version": "0.16.0",
        "openmc_commit": "617d35a5063c57796b43428bc401e627d2011046",
        "threads": 4,
        "source_domain_audit_sha256": "e" * 64,
        "source_point_cm": [0.5, 0.5, 0.5],
        "reference_source_mesh_sha256": "f" * 64,
        "reference_source_fingerprint": "1" * 64,
        "cross_sections_sha256": "2" * 64,
        "nuclear_data_manifest_sha256": "4" * 64,
        "model_xml_seed_normalized_fingerprint": "3" * 64,
        "histories": 4000,
        "batches": 2,
        "particles_per_batch": 2000,
        "timed_out": False,
        "exit_code": 0,
        "input_binding_pass": True,
        "transport_input_immutability_pass": True,
        "model_xml_immutability_pass": True,
        "log_qualification": parse_openmc_geometry_debug_log(
            CLEAN_LOG,
            exit_code=0,
            expected_threads=4,
            required_cell_ids=[1],
        ),
        "statepoint_summary_qualification": {
            "readable": True,
            "summary_geometry_present": True,
            "summary_contract_pass": True,
            "statepoint_batches": 2,
            "statepoint_particles": 2000,
            "statepoint_run_mode": "fixed source",
            "statepoint_seed": seed,
            "statepoint_seed_matches_expected": True,
        },
        "lost_particle_restart_files": [],
        "replica_gate_pass": True,
    }


def test_geometry_debug_aggregate_requires_two_unique_registered_seeds():
    passed = aggregate_geometry_debug_replicas(
        [_replica(20260827), _replica(20260828)],
        required_seeds=[20260827, 20260828],
    )
    duplicate = aggregate_geometry_debug_replicas(
        [_replica(20260827), _replica(20260827)],
        required_seeds=[20260827, 20260828],
    )
    missing = aggregate_geometry_debug_replicas(
        [_replica(20260827)], required_seeds=[20260827, 20260828]
    )

    assert passed["openmc_geometry_debug_gate_pass"] is True
    assert duplicate["openmc_geometry_debug_gate_pass"] is False
    assert missing["openmc_geometry_debug_gate_pass"] is False


def test_geometry_debug_aggregate_rejects_inconsistent_bindings():
    second = _replica(20260828)
    second["raw_h5m_sha256"] = "e" * 64
    report = aggregate_geometry_debug_replicas(
        [_replica(20260827), second],
        required_seeds=[20260827, 20260828],
    )

    assert report["bindings_pass"] is False
    assert report["openmc_geometry_debug_gate_pass"] is False


def test_geometry_debug_aggregate_rederives_replica_contract():
    forged = _replica(20260828)
    forged["statepoint_summary_qualification"]["statepoint_particles"] = 1
    report = aggregate_geometry_debug_replicas(
        [_replica(20260827), forged],
        required_seeds=[20260827, 20260828],
    )

    assert report["replica_contract_passes"] == [True, False]
    assert report["openmc_geometry_debug_gate_pass"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda log: log.update(completion_markers={}),
        lambda log: log.update(lost_or_navigation_diagnostics={"other": 0}),
        lambda log: log["overlap_check_coverage"].update(
            zero_check_required_cell_ids=[]
        )
        or log["overlap_check_coverage"]["rows"][0].update(overlap_checks=0),
        lambda log: log["overlap_check_coverage"].update(
            missing_required_cell_ids=[]
        )
        or log["overlap_check_coverage"].update(required_cell_ids=[1, 2]),
    ],
)
def test_geometry_debug_aggregate_rejects_forged_log_contract(mutation):
    forged = _replica(20260828)
    mutation(forged["log_qualification"])
    report = aggregate_geometry_debug_replicas(
        [_replica(20260827), forged],
        required_seeds=[20260827, 20260828],
    )

    assert report["replica_contract_passes"] == [True, False]
    assert report["openmc_geometry_debug_gate_pass"] is False
