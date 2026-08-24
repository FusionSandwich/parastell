"""Focused tests for deterministic PLC diagnostic evidence."""

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from parastell.native_port_geometry import (
    NativeSurfaceRecord,
    NativeVolumeRecord,
)
from parastell.plc_diagnostic import (
    CASE_MATRIX,
    classify_case,
    entity_ledger,
    exception_record,
    file_record,
    fresh_directory,
    inspect_edge_facet_intersections,
    serializable_intersection_report,
    terminal_classification,
    write_json,
    write_minimal_vtk,
)


def _synthetic_complex(*, intersect=True):
    target_x = 0.0 if intersect else 3.0
    surfaces = (
        NativeSurfaceRecord(
            "radial:inner",
            "radial_surface",
            np.asarray([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
            "inner",
            "outer",
        ),
        NativeSurfaceRecord(
            "radial:outer",
            "radial_surface",
            np.asarray(
                [
                    [
                        [target_x, -1.0, -1.0],
                        [target_x, 1.0, -1.0],
                        [target_x, 0.0, 1.0],
                    ]
                ]
            ),
            "outer",
            None,
        ),
    )
    return SimpleNamespace(
        surfaces=surfaces,
        volumes=(
            NativeVolumeRecord("inner", "layer", "A"),
            NativeVolumeRecord("outer", "layer", "B"),
        ),
        vertex_merge_tolerance=1.0e-9,
        radial_data={"names": ("inner", "outer")},
    )


def test_case_configuration_is_exact_and_deterministic():
    assert [case.to_dict() for case in CASE_MATRIX] == [
        {
            "name": "A0",
            "port": False,
            "num_ribs": 13,
            "num_rib_pts": 49,
            "resolution": "13x49",
        },
        {
            "name": "A1",
            "port": True,
            "num_ribs": 13,
            "num_rib_pts": 49,
            "resolution": "13x49",
        },
        {
            "name": "B0",
            "port": False,
            "num_ribs": 17,
            "num_rib_pts": 65,
            "resolution": "17x65",
        },
        {
            "name": "B1",
            "port": True,
            "num_ribs": 17,
            "num_rib_pts": 65,
            "resolution": "17x65",
        },
        {
            "name": "C0",
            "port": False,
            "num_ribs": 21,
            "num_rib_pts": 81,
            "resolution": "21x81",
        },
        {
            "name": "C1",
            "port": True,
            "num_ribs": 21,
            "num_rib_pts": 81,
            "resolution": "21x81",
        },
    ]


def test_output_directories_are_no_overwrite(tmp_path):
    output = fresh_directory(tmp_path / "A0")
    assert output.is_dir()
    with pytest.raises(FileExistsError):
        fresh_directory(output)


def test_file_record_hashes_exact_bytes(tmp_path):
    path = tmp_path / "input.bin"
    payload = b"ParaStell\x00PLC\n"
    path.write_bytes(payload)
    record = file_record(path)
    assert record["bytes"] == len(payload)
    assert record["sha256"] == sha256(payload).hexdigest()


def test_exception_capture_retains_type_message_and_traceback():
    try:
        raise RuntimeError("frozen failure")
    except RuntimeError as error:
        record = exception_record(error)
    assert record["type"] == "RuntimeError"
    assert record["message"] == "frozen failure"
    assert "raise RuntimeError" in record["traceback"]


def test_entity_ownership_and_known_intersection_are_reported():
    complex_ = _synthetic_complex(intersect=True)
    ledger = entity_ledger(complex_)
    report = inspect_edge_facet_intersections(complex_)
    assert len(ledger["facets"]) == 2
    assert report["intersection_count"] >= 1
    hit = report["intersections"][0]
    assert hit["edge"]["owners"]
    assert hit["edge"]["owners"][0]["normal"]
    assert hit["edge"]["owners"][0]["orientation"]
    assert hit["facet"]["owning_volumes"] == ["outer"]
    assert hit["relationship"] == "adjacent_radial_surfaces"
    assert hit["intersection_tolerance_cm"] == pytest.approx(1.0e-8)


def test_synthetic_no_intersection_control():
    report = inspect_edge_facet_intersections(
        _synthetic_complex(intersect=False)
    )
    assert report["intersection_count"] == 0


def test_minimal_json_and_vtk_export_only_implicated_facets(tmp_path):
    report = inspect_edge_facet_intersections(
        _synthetic_complex(intersect=True)
    )
    json_path = write_json(
        tmp_path / "intersection.json",
        serializable_intersection_report(report),
    )
    vtk_path = write_minimal_vtk(tmp_path / "implicated.vtk", report)
    assert '"_ledger"' not in json_path.read_text()
    vtk = vtk_path.read_text()
    assert "DATASET POLYDATA" in vtk
    assert "POLYGONS 2 8" in vtk


def test_unported_and_ported_classifications_remain_distinct():
    assert (
        classify_case(False, 1, None)
        == "BLOCKED_BASELINE_RADIAL_PLC_INTERSECTION"
    )
    assert (
        classify_case(True, 1, None) == "BLOCKED_PORT_STITCH_PLC_INTERSECTION"
    )
    combined = terminal_classification(
        [
            {
                "case": {"port": False},
                "intersection_count": 1,
                "downstream_error": None,
            },
            {
                "case": {"port": True},
                "intersection_count": 1,
                "downstream_error": None,
            },
        ]
    )
    assert combined == "BLOCKED_BASELINE_AND_PORTED_PLC_INTERSECTION"
