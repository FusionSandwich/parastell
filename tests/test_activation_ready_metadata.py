import json

import pytest

from parastell.activation_ready_metadata import SCHEMA
from parastell.activation_ready_metadata import (
    build_activation_ready_metadata,
)
from parastell.activation_ready_metadata import (
    read_activation_ready_metadata,
)
from parastell.activation_ready_metadata import (
    validate_activation_ready_metadata,
)
from parastell.activation_ready_metadata import (
    write_activation_ready_metadata,
)


MAGNET_ID = "machine-sector-coil-0001"


def inputs(*, nonoverlap_qualified=False):
    fingerprint = "1" * 64
    associations = {
        "schema": "parastell.magnet_associations/v1.0.0",
        "dagmc_sha256": "2" * 64,
        "canonical_geometry_fingerprint": fingerprint,
        "inventory": {
            "dagmc_geometry_sha256": "2" * 64,
            "magnet_pairs": [
                {
                    "magnet_id": MAGNET_ID,
                    "winding_pack": {
                        "volume_id": 101,
                        "material": "winding_pack",
                        "component_id": f"{MAGNET_ID}:winding_pack",
                        "magnet_id": MAGNET_ID,
                        "surface_ids": [13, 12],
                        "volume_cm3": 10.0,
                    },
                    "casing": {
                        "volume_id": 102,
                        "material": "magnet_casing",
                        "component_id": f"{MAGNET_ID}:casing",
                        "magnet_id": MAGNET_ID,
                        "surface_ids": [14, 15],
                        "volume_cm3": 5.0,
                    },
                }
            ],
        },
    }
    material_manifest = {
        "schema": "parastell.magnet_material_manifest/v1.0.0",
        "resolved_manifest_sha256": "3" * 64,
        "material_tags": {
            "winding_pack": "PackReference",
            "magnet_casing": "Steel",
        },
        "materials": {
            "PackReference": {
                "density_g_cm3": 8.5,
                "temperature_K": 293.6,
                "composition_basis": "mass_fraction",
                "nuclides": {"Cu63": 0.7, "Cu65": 0.3},
            },
            "Steel": {
                "density_g_cm3": 8.0,
                "temperature_K": 293.6,
                "composition_basis": "mass_fraction",
                "nuclides": {"Fe56": 1.0},
            },
        },
    }
    nuclear_data_manifest = {
        "schema": "parastell.nuclear_data_audit/v1.0.0",
        "cross_sections_xml": {
            "path": "/data/openmc/cross_sections.xml",
            "sha256": "4" * 64,
        },
        "evaluation_release": "fixture",
        "status": "PASS",
        "nuclides": [
            {"nuclide": "Cu63", "neutron_data_available": True},
            {"nuclide": "Cu65", "neutron_data_available": True},
            {"nuclide": "Fe56", "neutron_data_available": True},
        ],
    }
    local_mesh_manifest = {
        "schema": "parastell.magnet_local_mesh_collection/v1.0.0",
        "canonical_geometry_fingerprint": fingerprint,
        "cell_filter_applied": False,
        "nonoverlap_qualified": nonoverlap_qualified,
        "meshes": {
            MAGNET_ID: {
                "magnet_id": MAGNET_ID,
                "lower_left_local_cm": [0.0, 0.0, 0.0],
                "upper_right_local_cm": [4.0, 2.0, 2.0],
                "dimension": [2, 1, 1],
                "rotation_local_to_global": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "translation_global_cm": [10.0, 20.0, 30.0],
                "requested_resolution_cm": 2.0,
                "actual_bin_widths_cm": [2.0, 2.0, 2.0],
                "bin_count": 2,
                "bin_volume_cm3": 8.0,
                "frame_type": "coil_centerline_parallel_transport",
            }
        },
    }
    return {
        "associations": associations,
        "material_manifest": material_manifest,
        "nuclear_data_manifest": nuclear_data_manifest,
        "local_mesh_manifest": local_mesh_manifest,
        "openmc_cell_ids_by_dagmc_volume": {101: 501, 102: 502},
        "openmc_material_ids_by_tag": {
            "winding_pack": 41,
            "magnet_casing": 42,
        },
        "physical_source_rate_per_s": 2.4e20,
    }


def test_activation_metadata_preserves_ids_mass_hashes_and_mesh_caveat(
    tmp_path,
):
    artifact = build_activation_ready_metadata(**inputs())
    report = validate_activation_ready_metadata(artifact)

    assert artifact["schema"] == SCHEMA
    assert report["components"] == 2
    winding = next(
        item
        for item in artifact["components"]
        if item["component_role"] == "winding_pack"
    )
    assert winding["dagmc_volume_id"] == 101
    assert winding["openmc_cell_id"] == 501
    assert winding["material"]["openmc_material_id"] == 41
    assert winding["mass_g"] == pytest.approx(85.0)
    assert len(winding["material"]["isotopic_composition_sha256"]) == 64
    assert (
        artifact["nuclear_data"]["manifest"]["nuclides"][0]["nuclide"]
        == "Cu63"
    )
    assert len(artifact["nuclear_data"]["canonical_manifest_sha256"]) == 64

    mesh = artifact["local_meshes"][0]
    assert mesh["geometric_bin_volumes_cm3"] == [8.0, 8.0]
    assert (
        mesh["material_resolution"]["status"]
        == "MATERIAL_FRACTIONS_UNAVAILABLE"
    )
    assert mesh["material_resolution"]["materially_mixed_bins_possible"]
    assert not mesh["activation_readiness"]["direct_r2s"]["metadata_ready"]
    assert not mesh["activation_readiness"]["post_transport"][
        "post_transport_depletable"
    ]
    assert (
        mesh["activation_readiness"]["post_transport"]["status"]
        == "NOT_DEPLETABLE_MATERIAL_FRACTIONS_REQUIRED"
    )

    output = write_activation_ready_metadata(
        tmp_path / "activation.json", artifact
    )
    assert read_activation_ready_metadata(output) == artifact


def test_explicit_bin_fractions_and_nonoverlap_qualify_each_interface():
    arguments = inputs(nonoverlap_qualified=True)
    arguments["mesh_material_fractions_by_magnet"] = {
        MAGNET_ID: [
            {"winding_pack": 1.0},
            {"winding_pack": 0.25, "magnet_casing": 0.75},
        ]
    }
    artifact = build_activation_ready_metadata(**arguments)
    mesh = artifact["local_meshes"][0]

    assert mesh["activation_readiness"]["direct_r2s"]["metadata_ready"]
    assert mesh["activation_readiness"]["direct_r2s"]["nonoverlap_qualified"]
    assert mesh["activation_readiness"]["post_transport"][
        "post_transport_depletable"
    ]
    assert mesh["material_resolution"]["fractions_available"]
    assert mesh["material_resolution"]["materially_mixed_bins_possible"]
    assert artifact["activation_readiness"]["direct_r2s_mesh_metadata_ready"]
    assert artifact["activation_readiness"][
        "post_transport_mesh_metadata_ready"
    ]


def test_actual_openmc_ids_and_scalar_flux_semantics_are_mandatory():
    arguments = inputs()
    arguments["openmc_cell_ids_by_dagmc_volume"] = {101: 501}
    with pytest.raises(ValueError, match="actual OpenMC cell ID is missing"):
        build_activation_ready_metadata(**arguments)


def test_explicit_activation_metadata_requires_a_casing_without_breaking_base():
    arguments = inputs()
    arguments["associations"]["inventory"]["magnet_pairs"][0]["casing"] = None
    with pytest.raises(ValueError, match="lacks required casing metadata"):
        build_activation_ready_metadata(**arguments)

    arguments = inputs()
    arguments["flux_quantity"] = "surface_current"
    with pytest.raises(ValueError, match="surface current is not"):
        build_activation_ready_metadata(**arguments)


def test_reader_and_writer_reject_hash_tampering(tmp_path):
    artifact = build_activation_ready_metadata(**inputs())
    output = write_activation_ready_metadata(
        tmp_path / "activation.json", artifact
    )
    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["created_utc"] = "tampered"
    output.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        read_activation_ready_metadata(output)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        write_activation_ready_metadata(tmp_path / "copy.json", tampered)


def test_packaged_schema_identifies_the_same_strict_contract():
    schema_path = (
        __import__("parastell").__path__[0]
        + "/schemas/magnet_activation_ready_metadata.schema.json"
    )
    schema = json.loads(open(schema_path, encoding="utf-8").read())

    assert schema["properties"]["schema"]["const"] == SCHEMA
    assert schema["additionalProperties"] is False
    assert schema["properties"]["components"]["items"]["$ref"] == (
        "#/$defs/component"
    )
