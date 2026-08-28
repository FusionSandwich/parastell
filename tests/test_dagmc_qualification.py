import sys

from parastell.dagmc_qualification import audit_dagmc_topology
from parastell.dagmc_qualification import parse_check_watertight_log
from parastell.dagmc_qualification import parse_overlap_check_log


WATERTIGHT_PASS = """
number of surfaces=140
number of volumes=24
0/1234 (0.0%) unmatched edges
0/140 (0.0%) unsealed surfaces
0/24 (0.0%) unsealed volumes
leaky surface ids=
leaky volume ids=
"""


def _overlap_log(points, body):
    return (
        f"Checking {points} points along each triangle edge in addition to "
        f"the triangle vertices.\n{body}"
    )


def test_watertight_parser_requires_all_terminal_fields_and_clean_exit():
    passed = parse_check_watertight_log(WATERTIGHT_PASS, exit_code=0)
    missing = parse_check_watertight_log(
        "number of surfaces=140\n", exit_code=0
    )
    nonzero = parse_check_watertight_log(WATERTIGHT_PASS, exit_code=1)

    assert passed["pass"] is True
    assert missing["pass"] is False
    assert missing["missing_terminal_fields"]
    assert nonzero["pass"] is False


def test_watertight_parser_rejects_leaky_ids_and_inventory_mismatch():
    leaky = parse_check_watertight_log(
        WATERTIGHT_PASS.replace("leaky surface ids=", "leaky surface ids=7"),
        exit_code=0,
    )
    mismatch = parse_check_watertight_log(
        WATERTIGHT_PASS,
        exit_code=0,
        expected_surface_count=141,
        expected_volume_count=24,
    )

    assert leaky["leaky_id_lists_pass"] is False
    assert leaky["pass"] is False
    assert mismatch["native_inventory_count_reconciliation_pass"] is False
    assert mismatch["pass"] is False


def test_watertight_parser_rejects_inconsistent_or_zero_denominators():
    inconsistent = parse_check_watertight_log(
        WATERTIGHT_PASS.replace("0/140", "0/139"), exit_code=0
    )
    empty = parse_check_watertight_log(
        WATERTIGHT_PASS.replace(
            "number of volumes=24", "number of volumes=0"
        ).replace(
            "0/24 (0.0%) unsealed volumes", "0/0 (0.0%) unsealed volumes"
        ),
        exit_code=0,
    )

    assert inconsistent["denominator_reconciliation_pass"] is False
    assert inconsistent["pass"] is False
    assert empty["pass"] is False


def test_overlap_parser_accepts_only_zero_terminal_count_and_clean_exit():
    passed = parse_overlap_check_log(
        _overlap_log(4, "Overlap locations found: 0\n"),
        exit_code=0,
        points_per_edge=4,
        threads=4,
    )
    failed = parse_overlap_check_log(
        _overlap_log(
            4,
            "Overlap Location: 1.0 2.0 3.0\n"
            "Overlapping volumes: 4 5\n"
            "Overlap locations found: 1\n",
        ),
        exit_code=0,
        points_per_edge=4,
        threads=4,
    )

    assert passed["pass"] is True
    assert failed["pass"] is False
    assert failed["parsed_overlap_location_count"] == 1
    assert failed["locations"][0]["volume_ids"] == [4, 5]


def test_overlap_parser_accepts_native_no_overlap_sentence_only_once():
    passed = parse_overlap_check_log(
        _overlap_log(2, "No overlaps were found.\n"),
        exit_code=0,
        points_per_edge=2,
        threads=4,
    )
    ambiguous = parse_overlap_check_log(
        _overlap_log(
            2,
            "No overlaps were found.\nOverlap locations found: 0\n",
        ),
        exit_code=0,
        points_per_edge=2,
        threads=4,
    )

    assert passed["pass"] is True
    assert passed["terminal_overlap_location_count"] == 0
    assert passed["terminal_summary_kind"] == "no_overlaps_sentence"
    assert ambiguous["pass"] is False


def test_overlap_parser_fails_on_missing_or_inconsistent_terminal_count():
    missing = parse_overlap_check_log(
        _overlap_log(1, "no terminal summary\n"),
        exit_code=0,
        points_per_edge=1,
        threads=4,
    )
    inconsistent = parse_overlap_check_log(
        _overlap_log(
            1,
            "Overlap Location: 1 2 3\n"
            "Overlapping volumes: 4 5\n"
            "Overlap locations found: 0\n",
        ),
        exit_code=0,
        points_per_edge=1,
        threads=4,
    )

    assert missing["pass"] is False
    assert inconsistent["pass"] is False
    assert inconsistent["row_count_matches_terminal"] is False


def test_native_parsers_reject_concatenated_or_unparsed_summaries():
    duplicate_watertight = parse_check_watertight_log(
        WATERTIGHT_PASS + WATERTIGHT_PASS, exit_code=0
    )
    duplicate_overlap = parse_overlap_check_log(
        _overlap_log(
            2,
            "Overlap locations found: 0\nOverlap locations found: 1\n",
        ),
        exit_code=0,
        points_per_edge=2,
        threads=4,
    )
    unparsed_overlap = parse_overlap_check_log(
        _overlap_log(
            2,
            "Overlap Location: not-a-valid-row\n"
            "Overlap locations found: 0\n",
        ),
        exit_code=0,
        points_per_edge=2,
        threads=4,
    )

    assert duplicate_watertight["pass"] is False
    assert duplicate_overlap["pass"] is False
    assert duplicate_overlap["terminal_count_match_count"] == 2
    assert unparsed_overlap["pass"] is False
    assert unparsed_overlap["unparsed_overlap_token_count"] == 1


def test_overlap_parser_rejects_precision_header_mismatch():
    report = parse_overlap_check_log(
        _overlap_log(1, "Overlap locations found: 0\n"),
        exit_code=0,
        points_per_edge=4,
        threads=4,
    )

    assert report["precision_header_pass"] is False
    assert report["pass"] is False


def test_pydagmc_is_not_loaded_after_native_topology_failure(
    tmp_path, monkeypatch
):
    candidate = tmp_path / "candidate.h5m"
    candidate.write_bytes(b"not-loaded-by-the-failed-native-gate")

    monkeypatch.setattr(
        "parastell.native_dagmc_topology.audit_native_moab_topology",
        lambda *args, **kwargs: {
            "schema": "parastell.native_moab_topology/v1.0.0",
            "native_topology_gate_pass": False,
        },
    )
    monkeypatch.delitem(sys.modules, "pydagmc", raising=False)

    report = audit_dagmc_topology(candidate)

    assert report["topology_gate_pass"] is False
    assert report["pydagmc_load_attempted"] is False
    assert "pydagmc" not in sys.modules
