import json
from pathlib import Path

import numpy as np
import pytest

from parastell.coil_frame import parallel_transport_frame
from parastell.magnet_geometry_interchange import ARTIFACT_NAMES
from parastell.magnet_geometry_interchange import FRAME_KIND
from parastell.magnet_geometry_interchange import SCHEMA_URI
from parastell.magnet_geometry_interchange import (
    build_magnet_geometry_interchange,
)
from parastell.magnet_geometry_interchange import build_magnet_geometry_record
from parastell.magnet_geometry_interchange import (
    derive_boundary_geometry_fields,
)
from parastell.magnet_geometry_interchange import (
    read_magnet_geometry_interchange,
)
from parastell.magnet_geometry_interchange import (
    seal_magnet_geometry_interchange,
)
from parastell.magnet_geometry_interchange import (
    validate_magnet_geometry_interchange,
)
from parastell.magnet_geometry_interchange import (
    write_magnet_geometry_interchange,
)


def _frame():
    angle = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    points = np.column_stack(
        (10.0 * np.cos(angle), 10.0 * np.sin(angle), np.zeros_like(angle))
    )
    return parallel_transport_frame(points, closed=True)


def _artifacts():
    return {
        "outer_step": {
            "status": "available",
            "sha256": "1" * 64,
            "size_bytes": 101,
        },
        "outer_stl": {
            "status": "unavailable",
            "reason": "STL was not exported by this pilot",
        },
        "outer_h5m": {
            "status": "available",
            "sha256": "2" * 64,
            "size_bytes": 202,
        },
        "winding_pack_step": {
            "status": "available",
            "sha256": "3" * 64,
            "size_bytes": 303,
        },
        "winding_pack_stl": {
            "status": "unavailable",
            "reason": "No scientifically qualified winding-pack STL exists",
        },
    }


def _magnet():
    return build_magnet_geometry_record(
        magnet_id="coil-0001",
        coil_filament_source_sha256="a" * 64,
        source_filament_closed=True,
        represented_segment_kind="full_filament",
        centreline_frame=_frame(),
        native_global_transform=np.eye(4),
        field_period_index=0,
        sector_index=0,
        field_period_sector_transform=np.eye(4),
        outer_casing_cross_section_local_cm=[
            [-2.0, -1.5],
            [2.0, -1.5],
            [2.0, 1.5],
            [-2.0, 1.5],
        ],
        outer_winding_pack_cross_section_local_cm=[
            [-1.7, -1.2],
            [1.7, -1.2],
            [1.7, 1.2],
            [-1.7, 1.2],
        ],
        casing_thickness_cm=0.3,
        artifacts=_artifacts(),
        surface_ids=[101, 102, 103],
        facet_ids=["facet-a", "facet-b"],
        components=[
            {"component_name": "outer_casing", "material_name": "SS316L"},
            {
                "component_name": "winding_pack",
                "material_name": "WindingPackReference",
            },
        ],
    )


def _interchange():
    return build_magnet_geometry_interchange(
        [_magnet()],
        geometry_fingerprint="b" * 64,
        provenance={"parastell_commit": "c" * 40},
    )


def test_geometry_interchange_is_deterministic_and_round_trips(tmp_path):
    first = _interchange()
    second = _interchange()
    assert first == second
    assert first["schema"] == SCHEMA_URI
    magnet = first["magnets"][0]
    assert magnet["frame"]["frame_kind"] == FRAME_KIND
    assert magnet["frame"]["tape_twist_resolved"] is False
    assert magnet["closure"] == {
        "source_filament_closed": True,
        "represented_segment_closed": True,
        "represented_segment_kind": "full_filament",
    }
    output = tmp_path / "magnet-geometry-interchange.json"
    write_magnet_geometry_interchange(output, first)
    restored = read_magnet_geometry_interchange(output)
    assert restored == first
    duplicate = tmp_path / "duplicate.json"
    write_magnet_geometry_interchange(duplicate, restored)
    assert duplicate.read_bytes() == output.read_bytes()


def test_engineering_frame_samples_are_right_handed_and_continuous():
    frame = _magnet()["frame"]
    widths = []
    for sample in frame["samples"]:
        tangent = np.asarray(sample["tangent_global"])
        width = np.asarray(sample["width_direction_global"])
        normal = np.asarray(sample["normal_direction_global"])
        assert np.allclose(np.cross(tangent, width), normal)
        widths.append(width)
    widths = np.asarray(widths)
    assert np.all(np.sum(widths[:-1] * widths[1:], axis=1) > 0.0)
    assert np.dot(widths[-1], widths[0]) > 0.0

    invalid = _interchange()
    invalid["magnets"][0]["frame"]["samples"][0]["normal_direction_global"] = [
        0.0,
        0.0,
        -1.0,
    ]
    with pytest.raises(ValueError, match="right-handed orthonormal"):
        seal_magnet_geometry_interchange(invalid)


def test_geometry_contract_rejects_false_arclength_and_tangent_claims():
    invalid_arc = _interchange()
    invalid_arc["magnets"][0]["centreline"]["arc_length_cm"][1] += 0.25
    with pytest.raises(ValueError, match="arc_length_cm disagrees"):
        seal_magnet_geometry_interchange(invalid_arc)

    invalid_total = _interchange()
    invalid_total["magnets"][0]["centreline"]["total_length_cm"] += 0.25
    with pytest.raises(ValueError, match="total_length_cm disagrees"):
        seal_magnet_geometry_interchange(invalid_total)

    invalid_tangent = _interchange()
    sample = invalid_tangent["magnets"][0]["frame"]["samples"][0]
    sample["tangent_global"] = [1.0, 0.0, 0.0]
    sample["normal_direction_global"] = np.cross(
        sample["tangent_global"], sample["width_direction_global"]
    ).tolist()
    with pytest.raises(ValueError, match="tangent samples disagree"):
        seal_magnet_geometry_interchange(invalid_tangent)


def test_read_rejects_tampered_hash_bound_content(tmp_path):
    output = tmp_path / "geometry.json"
    write_magnet_geometry_interchange(output, _interchange())
    document = json.loads(output.read_text(encoding="utf-8"))
    document["magnets"][0]["magnet_id"] = "tampered-coil"
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="content SHA-256 mismatch"):
        read_magnet_geometry_interchange(output)


def test_unavailable_artifacts_are_explicit_and_cannot_carry_fake_hashes():
    magnet = _magnet()
    assert set(magnet["artifacts"]) == set(ARTIFACT_NAMES)
    unavailable = magnet["artifacts"]["outer_stl"]
    assert unavailable == {
        "status": "unavailable",
        "reason": "STL was not exported by this pilot",
    }
    invalid = _interchange()
    invalid["magnets"][0]["artifacts"]["outer_stl"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unexpected.*sha256"):
        seal_magnet_geometry_interchange(invalid)


def test_validator_is_strict_and_schema_file_tracks_contract():
    invalid = _interchange()
    invalid["magnets"][0]["implementation_object"] = "CadQuery.Shape"
    with pytest.raises(ValueError, match="unexpected.*implementation_object"):
        seal_magnet_geometry_interchange(invalid)

    schema_path = (
        Path(__file__).parents[1]
        / "parastell"
        / "schemas"
        / "magnet_geometry_interchange.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["schema"]["const"] == SCHEMA_URI
    assert schema["additionalProperties"] is False


def test_boundary_helper_derives_facet_and_local_frame_fields():
    frame = _frame()
    point = frame.points_cm[0]
    result = derive_boundary_geometry_fields(
        {"position_global_cm": point.tolist(), "facet_id": "facet-at-s0"},
        centreline_frame=frame,
        facet_vertices_global_cm=[
            point.tolist(),
            (point + [0.0, 1.0, 0.0]).tolist(),
            (point + [0.0, 0.0, 1.0]).tolist(),
        ],
    )
    assert result["facet_id"] == "facet-at-s0"
    assert result["barycentric_coordinates"] == pytest.approx([1.0, 0.0, 0.0])
    assert result["s_cm"] == pytest.approx(0.0)
    assert result["local_cross_section_coordinates_cm"] == pytest.approx(
        [0.0, 0.0]
    )
    assert result["frame_kind"] == FRAME_KIND
    assert result["tape_twist_resolved"] is False
    assert np.allclose(
        np.cross(result["local_tangent"], result["local_width_direction"]),
        result["local_normal_direction"],
    )


def test_boundary_helper_never_fabricates_a_facet_identity():
    with pytest.raises(ValueError, match="facet_id is required"):
        derive_boundary_geometry_fields(
            {"position_global_cm": _frame().points_cm[0].tolist()},
            centreline_frame=_frame(),
            facet_vertices_global_cm=[
                [10.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
                [10.0, 0.0, 1.0],
            ],
        )


def test_open_represented_segment_is_distinct_from_closed_source_filament():
    angle = np.linspace(0.0, np.pi / 2.0, 8)
    frame = parallel_transport_frame(
        np.column_stack(
            (
                10.0 * np.cos(angle),
                10.0 * np.sin(angle),
                np.zeros_like(angle),
            )
        ),
        closed=False,
    )
    magnet = build_magnet_geometry_record(
        magnet_id="coil-segment",
        coil_filament_source_sha256="d" * 64,
        source_filament_closed=True,
        represented_segment_kind="field_period_segment",
        centreline_frame=frame,
        native_global_transform=np.eye(4),
        field_period_index=1,
        sector_index=2,
        field_period_sector_transform=np.eye(4),
        outer_casing_cross_section_local_cm=[
            [-2, -1],
            [2, -1],
            [2, 1],
            [-2, 1],
        ],
        outer_winding_pack_cross_section_local_cm=[
            [-1.5, -0.5],
            [1.5, -0.5],
            [1.5, 0.5],
            [-1.5, 0.5],
        ],
        casing_thickness_cm=0.5,
        artifacts=_artifacts(),
        surface_ids=[201],
        facet_ids=[],
        components=[
            {"component_name": "outer_casing", "material_name": "SS316L"},
            {
                "component_name": "winding_pack",
                "material_name": "WindingPackReference",
            },
        ],
    )
    assert magnet["closure"] == {
        "source_filament_closed": True,
        "represented_segment_closed": False,
        "represented_segment_kind": "field_period_segment",
    }
    validate_magnet_geometry_interchange(
        build_magnet_geometry_interchange(
            [magnet], geometry_fingerprint="e" * 64, provenance={}
        )
    )
