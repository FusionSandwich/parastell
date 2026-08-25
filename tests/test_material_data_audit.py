import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from parastell.material_manifest import audit_nuclear_data
from parastell.material_manifest import resolve_material_manifest


FUSION_MATERIAL_DB_COMMIT = "9d8f84ad7b3ac2c587edfbe8ec1ed74891484498"
PURE_MATERIAL_ARTIFACT_SHA256 = (
    "bb15bbad56395269b2e07df183d35fa37894dd5f752ebc793166566d40d61577"
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_openmc_tables(tmp_path, maximum_energy_eV=20.0e6):
    neutron = tmp_path / "C12.h5"
    with h5py.File(neutron, "w") as data:
        nuclide = data.create_group("C12")
        energy = nuclide.create_group("energy")
        energy.create_dataset(
            "300K", data=np.asarray([1.0e-5, maximum_energy_eV])
        )
        kts = nuclide.create_group("kTs")
        kts.create_dataset("300K", data=0.025852)
        reactions = nuclide.create_group("reactions")
        elastic = reactions.create_group("reaction_002")
        elastic.attrs["mt"] = 2
        elastic.attrs["label"] = np.bytes_(b"(n,elastic)")
        capture = reactions.create_group("reaction_102")
        capture.attrs["mt"] = 102
        capture.attrs["label"] = np.bytes_(b"(n,gamma)")

    photon = tmp_path / "C.h5"
    with h5py.File(photon, "w") as data:
        element = data.create_group("C")
        element.create_dataset(
            "energy", data=np.asarray([1.0e3, maximum_energy_eV])
        )

    catalog = tmp_path / "cross_sections.xml"
    catalog.write_text(
        "<cross_sections>\n"
        '  <library materials="C12" path="C12.h5" type="neutron" />\n'
        '  <library materials="C" path="C.h5" type="photon" />\n'
        "</cross_sections>\n",
        encoding="utf-8",
    )
    return catalog, neutron, photon


def _audit_manifest():
    return {
        "materials": {
            "Eins": {
                "temperature_K": 300.0,
                "nuclides": {"C12": 1.0},
            }
        }
    }


def _run_audit(catalog, **overrides):
    arguments = {
        "cross_sections_path": catalog,
        "approved_library": "fixture-evaluation",
        "evaluation_release": "fixture-v1",
        "photon_evaluation_release": "fixture-v1",
        "requested_source_max_energy_eV": 14.1e6,
        "temperature_method": "nearest",
        "temperature_tolerance_K": 1.0,
    }
    arguments.update(overrides)
    return audit_nuclear_data(_audit_manifest(), **arguments)


def test_production_configuration_freezes_pinned_eins_insulation():
    config_path = (
        Path(__file__).parents[1]
        / "examples"
        / "magnet_radiation_materials.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = config["generated_artifacts"][0]
    selection = config["production_material_selection"]
    reference = config["winding_pack_variants"]["WindingPackReference"]

    assert config["fusion_material_db"]["commit"] == FUSION_MATERIAL_DB_COMMIT
    assert source["sha256"] == PURE_MATERIAL_ARTIFACT_SHA256
    assert "Eins" in source["selected_materials"]
    assert selection == {
        "winding_pack": "WindingPackReference",
        "insulation": "Eins",
        "insulation_source_repository": "FusionSandwich/fusion-material-db",
        "insulation_source_commit": FUSION_MATERIAL_DB_COMMIT,
        "insulation_source_artifact": (
            "db-outputs/PureFusionMaterials_libv1.json"
        ),
        "insulation_source_artifact_sha256": PURE_MATERIAL_ARTIFACT_SHA256,
        "insulation_density_g_cm3": 1.8,
        "insulation_citation": "FESS-FNSF and ARIES GFFpolyimide",
    }
    assert reference["production_selected"] is True
    assert reference["designated_insulation_material"] == "Eins"
    assert reference["constituent_volume_fractions"]["Eins"] == 0.08
    assert "InsulationCarbon" not in reference["constituent_volume_fractions"]
    assert "InsulationCarbon" not in config["explicit_materials"]
    assert all(
        variant["designated_insulation_material"] == "Eins"
        and "Eins" in variant["constituent_volume_fractions"]
        for variant in config["winding_pack_variants"].values()
    )


def test_resolved_generated_record_retains_hash_bound_provenance(tmp_path):
    generated = tmp_path / "PureFusionMaterials_libv1.json"
    generated.write_text(
        json.dumps(
            {
                "Eins": {
                    "density": 1.8,
                    "comp": {"C12": 1.0},
                    "metadata": {
                        "citation": "FESS-FNSF and ARIES GFFpolyimide",
                        "name": "Eins",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    artifact_hash = _sha256(generated)
    configuration = tmp_path / "materials.json"
    configuration.write_text(
        json.dumps(
            {
                "fusion_material_db": {
                    "repository": "FusionSandwich/fusion-material-db",
                    "commit": FUSION_MATERIAL_DB_COMMIT,
                },
                "generated_artifacts": [
                    {
                        "path": str(generated),
                        "sha256": artifact_hash,
                        "selected_materials": ["Eins"],
                    }
                ],
                "winding_pack_variants": {
                    "Pack": {
                        "fraction_basis": "volume",
                        "production_selected": True,
                        "designated_insulation_material": "Eins",
                        "constituent_volume_fractions": {"Eins": 1.0},
                    }
                },
                "production_material_selection": {
                    "winding_pack": "Pack",
                    "insulation": "Eins",
                    "insulation_density_g_cm3": 1.8,
                },
                "material_tags": {"winding_pack": "Pack"},
            }
        ),
        encoding="utf-8",
    )

    manifest = resolve_material_manifest(configuration)
    insulation = manifest["materials"]["Eins"]
    provenance = insulation["source_provenance"]
    assert insulation["density_g_cm3"] == 1.8
    assert insulation["metadata"]["citation"] == (
        "FESS-FNSF and ARIES GFFpolyimide"
    )
    assert provenance["repository_commit"] == FUSION_MATERIAL_DB_COMMIT
    assert provenance["artifact_sha256"] == artifact_hash
    assert manifest["production_material_selection"]["insulation"] == "Eins"
    assert (
        manifest["materials"]["Pack"]["homogenization"][
            "designated_insulation_material"
        ]
        == "Eins"
    )


def test_audit_hashes_tables_and_reads_energy_reactions_and_temperature(
    tmp_path,
):
    catalog, neutron, photon = _write_openmc_tables(tmp_path)
    audit = _run_audit(catalog)
    row = audit["nuclides"][0]

    assert audit["status"] == "PASS"
    assert audit["evaluation_classification"] == "SINGLE_EVALUATION"
    assert audit["requested_source_energy"] == {
        "maximum_eV": 14.1e6,
        "declared": True,
        "all_neutron_tables_support_range": True,
    }
    assert row["neutron_table"]["sha256"] == _sha256(neutron)
    assert row["photon_table"]["sha256"] == _sha256(photon)
    assert row["incident_energy_support"]["actual_maximum_eV"] == 20.0e6
    assert row["incident_energy_support"]["requested_source_range_supported"]
    assert row["temperature_support"]["available_K"] == [300.0]
    assert row["reaction_mt_labels"] == [
        {"mt": 2, "label": "(n,elastic)"},
        {"mt": 102, "label": "(n,gamma)"},
    ]
    assert audit["material_coverage"][0]["status"] == "PASS"


def test_audit_fails_closed_when_fusion_source_exceeds_table_domain(tmp_path):
    catalog, _, _ = _write_openmc_tables(tmp_path, maximum_energy_eV=10.0e6)
    audit = _run_audit(catalog)
    assert audit["status"] == "FAIL_SOURCE_ENERGY_RANGE"
    assert audit["unsupported_source_energy_nuclides"] == ["C12"]


def test_audit_classifies_mixed_evaluations_explicitly(tmp_path):
    catalog, _, _ = _write_openmc_tables(tmp_path)
    rejected = _run_audit(
        catalog,
        photon_evaluation_release="fixture-photon-v2",
        approved_mixed_case=False,
    )
    assert rejected["status"] == "FAIL_UNAPPROVED_MIXED_LIBRARY"
    assert rejected["evaluation_classification"] == (
        "UNAPPROVED_MIXED_EVALUATION"
    )

    accepted = _run_audit(
        catalog,
        photon_evaluation_release="fixture-photon-v2",
        approved_mixed_case=True,
    )
    assert accepted["status"] == "PASS"
    assert accepted["evaluation_classification"] == (
        "EXPLICIT_APPROVED_MIXED_EVALUATION"
    )
    assert accepted["evaluation_policy"]["mixed_data_sensitivity_required"]


@pytest.mark.parametrize("maximum", [0.0, -1.0, float("nan")])
def test_audit_rejects_invalid_requested_source_domain(tmp_path, maximum):
    catalog, _, _ = _write_openmc_tables(tmp_path)
    with pytest.raises(ValueError, match="finite and positive"):
        _run_audit(catalog, requested_source_max_energy_eV=maximum)
