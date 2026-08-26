import json

import pytest

from parastell.material_intersection_volumes import (
    build_material_intersection_volume_manifest,
    read_material_intersection_volume_manifest,
    write_material_intersection_volume_manifest,
)

MAGNET_ID = "machine-sector-coil-0005"


def _inputs():
    fingerprint = "1" * 64
    local_mesh_manifest = {
        "schema": "parastell.magnet_local_mesh_collection/v1.0.0",
        "canonical_geometry_fingerprint": fingerprint,
        "cell_filter_applied": True,
        "meshes": {
            MAGNET_ID: {
                "magnet_id": MAGNET_ID,
                "lower_left_local_cm": [0.0, 0.0, 0.0],
                "upper_right_local_cm": [2.0, 1.0, 1.0],
                "dimension": [2, 1, 1],
                "rotation_local_to_global": [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "translation_global_cm": [10.0, 20.0, 30.0],
                "requested_resolution_cm": 1.0,
                "actual_bin_widths_cm": [1.0, 1.0, 1.0],
                "bin_count": 2,
                "bin_volume_cm3": 1.0,
                "frame_type": "coil_centerline_parallel_transport",
            }
        },
    }
    associations = {
        "canonical_geometry_fingerprint": fingerprint,
        "inventory": {
            "dagmc_geometry_sha256": "2" * 64,
            "magnet_pairs": [
                {
                    "magnet_id": MAGNET_ID,
                    "winding_pack": {
                        "volume_id": 18,
                        "material": "winding_pack",
                    },
                }
            ],
        },
    }
    return {
        "local_mesh_manifest": local_mesh_manifest,
        "associations": associations,
        "openmc_cell_ids_by_dagmc_volume": {18: 118},
        "openmc_material_ids_by_tag": {"winding_pack": 41},
        "results_by_magnet": {
            MAGNET_ID: {
                "intersection_volume_cm3": [0.25, 0.0],
                "standard_deviation_cm3": [0.01, 0.0],
            }
        },
        "method": "qualified_rotated_mesh_component_raytrace",
        "estimator_provenance_sha256": "3" * 64,
        "coordinate_transform_qualification": {
            "status": "PASS",
            "rotation_applied": True,
            "translation_applied": True,
            "fixture": "curved_surface_frame_mapping",
            "evidence_sha256": "4" * 64,
        },
        "maximum_relative_standard_deviation": 0.05,
    }


def test_component_intersection_volumes_round_trip_and_bind_identities(
    tmp_path,
):
    arguments = _inputs()
    artifact = build_material_intersection_volume_manifest(**arguments)
    row = artifact["meshes"][MAGNET_ID]

    assert artifact["status"] == "QUALIFIED"
    assert row["openmc_cell_id"] == 118
    assert row["material_tag"] == "winding_pack"
    assert row["bin_status"] == ["QUALIFIED", "EMPTY_INTERSECTION"]
    assert row["total_intersection_volume_cm3"] == pytest.approx(0.25)

    output = tmp_path / "material_intersection_volumes.json"
    write_material_intersection_volume_manifest(output, artifact)
    loaded = read_material_intersection_volume_manifest(
        output,
        local_mesh_manifest=arguments["local_mesh_manifest"],
        associations=arguments["associations"],
    )
    assert loaded == artifact


def test_full_voxel_substitution_and_unqualified_rotation_fail_closed():
    arguments = _inputs()
    arguments["results_by_magnet"][MAGNET_ID]["intersection_volume_cm3"][
        0
    ] = 1.01
    with pytest.raises(ValueError, match="exceeds geometric bin volume"):
        build_material_intersection_volume_manifest(**arguments)

    arguments = _inputs()
    arguments["coordinate_transform_qualification"]["rotation_applied"] = False
    with pytest.raises(ValueError, match="rotation and translation"):
        build_material_intersection_volume_manifest(**arguments)


def test_reader_rejects_hash_tampering(tmp_path):
    artifact = build_material_intersection_volume_manifest(**_inputs())
    output = tmp_path / "material_intersection_volumes.json"
    write_material_intersection_volume_manifest(output, artifact)
    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["meshes"][MAGNET_ID]["intersection_volume_cm3"][0] = 0.5
    output.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        read_material_intersection_volume_manifest(output)


def test_explicit_selected_magnet_subset_preserves_full_input_binding(
    tmp_path,
):
    arguments = _inputs()
    other = "machine-sector-coil-0006"
    arguments["local_mesh_manifest"]["meshes"][other] = {
        **arguments["local_mesh_manifest"]["meshes"][MAGNET_ID],
        "magnet_id": other,
    }
    arguments["associations"]["inventory"]["magnet_pairs"].append(
        {
            "magnet_id": other,
            "winding_pack": {"volume_id": 19, "material": "winding_pack"},
        }
    )
    arguments["selected_magnet_ids"] = [MAGNET_ID]

    artifact = build_material_intersection_volume_manifest(**arguments)
    assert artifact["selected_magnet_ids"] == [MAGNET_ID]
    assert artifact["selection_scope"] == "explicit_selected_magnets"
    assert list(artifact["meshes"]) == [MAGNET_ID]

    output = tmp_path / "selected_material_intersection_volumes.json"
    write_material_intersection_volume_manifest(output, artifact)
    loaded = read_material_intersection_volume_manifest(
        output,
        local_mesh_manifest=arguments["local_mesh_manifest"],
        associations=arguments["associations"],
    )
    assert loaded == artifact
