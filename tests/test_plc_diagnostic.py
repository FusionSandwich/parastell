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
    _port_distances,
    classify_case,
    entity_ledger,
    exception_record,
    file_record,
    fresh_directory,
    inspect_edge_facet_intersections,
    process_failure_record,
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
            "radial_diagonal": "sector_edge_alternate",
            "resolution": "13x49",
        },
        {
            "name": "A1",
            "port": True,
            "num_ribs": 13,
            "num_rib_pts": 49,
            "radial_diagonal": "sector_edge_alternate",
            "resolution": "13x49",
        },
        {
            "name": "B0",
            "port": False,
            "num_ribs": 17,
            "num_rib_pts": 65,
            "radial_diagonal": "primary",
            "resolution": "17x65",
        },
        {
            "name": "B1",
            "port": True,
            "num_ribs": 17,
            "num_rib_pts": 65,
            "radial_diagonal": "alternate",
            "resolution": "17x65",
        },
        {
            "name": "C0",
            "port": False,
            "num_ribs": 21,
            "num_rib_pts": 81,
            "radial_diagonal": "alternate",
            "resolution": "21x81",
        },
        {
            "name": "C1",
            "port": True,
            "num_ribs": 21,
            "num_rib_pts": 81,
            "radial_diagonal": "primary",
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


def test_native_process_signal_is_captured_without_losing_identity():
    record = process_failure_record(-11, "native stderr")
    assert record == {
        "type": "ProcessSignalFailure",
        "message": (
            "Diagnostic child terminated by signal 11 during Gmsh discrete "
            "PLC tetrahedralization"
        ),
        "traceback": "native stderr",
        "return_code": -11,
    }


def test_unstitched_comparison_grid_reports_patch_boundary_distance():
    class Surface:
        @staticmethod
        def evaluate(phi, theta):
            return np.asarray((phi, theta, 0.0))

    values = np.asarray((-1.8, 0.0, 1.8))
    grid = np.asarray(
        [[Surface.evaluate(phi, theta) for theta in values] for phi in values]
    )
    loop = SimpleNamespace(
        inner_points=np.asarray(
            (
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (1.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
                (-1.0, -1.0, 0.0),
            )
        ),
        outer_points=np.asarray(
            (
                (-1.2, -1.2, 0.0),
                (1.2, -1.2, 0.0),
                (1.2, 1.2, 0.0),
                (-1.2, 1.2, 0.0),
                (-1.2, -1.2, 0.0),
            )
        ),
    )
    source_model = SimpleNamespace(
        native_radial_stack=lambda: (("inner", Surface()),),
        _port_aperture_half_width=lambda _port: 1.0,
    )
    complex_ = SimpleNamespace(
        loops=(loop,),
        radial_data={
            "grids": [grid],
            "phi_values": values,
            "theta_values": values,
            "radial_loop_indices": [None],
        },
        port=SimpleNamespace(
            placement=SimpleNamespace(
                surface_anchor=SimpleNamespace(
                    toroidal_angle=0.0, poloidal_angle=0.0
                )
            )
        ),
        source_model=source_model,
    )
    distances = _port_distances(complex_, np.zeros(3))
    assert distances["aperture_boundary_cm"] == pytest.approx(1.0)
    assert distances["liner_boundary_cm"] == pytest.approx(1.2)
    assert distances["patch_boundary_cm"] == pytest.approx(1.8)


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


def _valid_matrix_payloads():
    regions = {"plasma": 2, "blanket": 3}
    mesh_audit = {
        "region_tetrahedron_counts": regions,
        "region_minimum_tetrahedron_volume": {name: 1.0 for name in regions},
        "region_maximum_tetrahedron_volume": {name: 1.0 for name in regions},
        "region_total_tetrahedron_volume": {name: float(count) for name, count in regions.items()},
        "tetrahedron_count": 5,
        "inverted_tetrahedron_count": 0,
        "zero_volume_tetrahedron_count": 0,
        "duplicate_tetrahedron_count": 0,
        "nonconformal_interface_face_count": 0,
        "disconnected_region_count": 0,
        "region_reference_volume": {name: 1.0 for name in regions},
        "region_relative_volume_error": {name: 0.0 for name in regions},
        "region_minimum_scaled_jacobian": {name: 1.0 for name in regions},
        "region_minimum_mean_ratio": {name: 1.0 for name in regions},
        "region_minimum_radius_ratio": {name: 1.0 for name in regions},
        "region_minimum_dihedral_angle": {name: 1.0 for name in regions},
        "region_maximum_dihedral_angle": {name: 1.0 for name in regions},
        "region_minimum_edge_length": {name: 1.0 for name in regions},
        "region_maximum_edge_length": {name: 1.0 for name in regions},
        "region_quality_threshold_counts": {
            name: {"scaled_jacobian": 0} for name in regions
        },
        "quality_thresholds": {"scaled_jacobian": 0.1},
    }
    return [
        {
            "schema_version": "1.0",
            "scope": "PLC/topology diagnostic; not qualified transport geometry",
            "case": case.to_dict(),
            "port_geometry_present": case.port,
            "radial_diagonal": case.radial_diagonal,
            "repository_sha": "expected-sha",
            "process_return_code": 0,
            "passed": True,
            "classification": "PASS_NO_REPRODUCIBLE_PLC_FAILURE",
            "intersection_count": 0,
            "downstream_error": None,
            "mesh_validation": dict(mesh_audit),
        }
        for case in CASE_MATRIX
    ]


def test_terminal_pass_requires_valid_complete_child_results():
    assert (
        terminal_classification(_valid_matrix_payloads(), "expected-sha")
        == "PASS_NO_REPRODUCIBLE_PLC_FAILURE"
    )


@pytest.mark.parametrize("invalid_region_name", ("", "   "))
def test_blank_region_names_cannot_pass(invalid_region_name):
    rows = _valid_matrix_payloads()
    for row in rows:
        audit = row["mesh_validation"]
        for field in audit:
            if field.startswith("region_"):
                values = audit[field]
                audit[field] = {invalid_region_name: next(iter(values.values()))}
        audit["region_tetrahedron_counts"] = {invalid_region_name: 5}
        audit["tetrahedron_count"] = 5
    assert (
        terminal_classification(rows, "expected-sha")
        == "BLOCKED_ENVIRONMENT_OR_INPUT_IDENTITY"
    )


def test_boolean_process_return_code_cannot_pass():
    rows = _valid_matrix_payloads()
    rows[0]["process_return_code"] = False
    assert (
        terminal_classification(rows, "expected-sha")
        == "BLOCKED_ENVIRONMENT_OR_INPUT_IDENTITY"
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda rows: rows[0].update(repository_sha="stale-sha"),
        lambda rows: rows[0].update(case={**rows[0]["case"], "name": "B0"}),
        lambda rows: rows.pop(),
        lambda rows: rows.__setitem__(5, dict(rows[4])),
        lambda rows: rows[0].update(process_return_code=2),
        lambda rows: rows[0].pop("downstream_error"),
        lambda rows: rows[0].update(mesh_validation=None),
        lambda rows: rows[0]["mesh_validation"].update(tetrahedron_count=0),
    ),
    ids=(
        "wrong-hash",
        "wrong-identity",
        "missing-case",
        "duplicate-case",
        "failed-child",
        "missing-downstream-status",
        "missing-mesh",
        "bad-mesh-count",
    ),
)
def test_malformed_or_failed_matrix_cannot_pass(mutate):
    rows = _valid_matrix_payloads()
    mutate(rows)
    assert (
        terminal_classification(rows, "expected-sha")
        == "BLOCKED_ENVIRONMENT_OR_INPUT_IDENTITY"
    )


@pytest.mark.parametrize(
    ("case_index", "expected"),
    (
        (0, "BLOCKED_BASELINE_RADIAL_PLC_INTERSECTION"),
        (1, "BLOCKED_PORT_STITCH_PLC_INTERSECTION"),
    ),
)
def test_intersections_remain_blocked_even_without_complete_mesh_audit(
    case_index, expected
):
    rows = _valid_matrix_payloads()
    rows[case_index]["intersection_count"] = 1
    rows[case_index]["mesh_validation"] = None
    assert terminal_classification(rows, "expected-sha") == expected


def test_original_zero_crossing_counterexample_fails_closed():
    rows = _valid_matrix_payloads()
    for row in rows:
        row.pop("mesh_validation")
        row["passed"] = False
    assert (
        terminal_classification(rows, "expected-sha")
        == "BLOCKED_ENVIRONMENT_OR_INPUT_IDENTITY"
    )
