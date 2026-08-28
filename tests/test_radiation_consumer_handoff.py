from __future__ import annotations

import copy
from pathlib import Path

import pytest

from parastell.downstream_response_export import (
    build_downstream_exports,
    write_downstream_exports,
)
from parastell.radiation_consumer_handoff import (
    PHASE_SPACE_FIELDS,
    RadiationHandoffError,
    build_activation_schedule_reference,
    build_radiation_consumer_handoff,
    spectra_pka_inputs,
    validate_radiation_consumer_handoff,
)


def _provenance():
    return {
        "raw_h5m_sha256": "a" * 64,
        "canonical_geometry_fingerprint": "b" * 64,
        "source_definition_sha256": "c" * 64,
        "statepoint_sha256": "d" * 64,
        "nuclear_data_sha256": "e" * 64,
        "model_xml_sha256": "5" * 64,
        "settings_xml_sha256": "6" * 64,
        "strict_run_audit_sha256": "7" * 64,
        "root_acceptance_receipt_sha256": "8" * 64,
        "nuclear_data_library": "ENDF/B-VIII.0",
        "openmc_version": "0.16.0",
        "source_histories": 1000,
        "physical_source_rate_per_s": 1.0e20,
        "source_rate_scope": "modeled_period",
        "normalization": "per_source_history",
        "statistics_classification": "WORKFLOW_SMOKE_ONLY",
    }


def _materials():
    return [
        {
            "domain_id": "magnet-0000",
            "material_id": "homogenized-magnet",
            "composition_sha256": "f" * 64,
            "volume_cm3": 12.0,
            "openmc_cell_id": 9001,
            "dagmc_volume_id": 9,
            "volume_receipt_sha256": "9" * 64,
            "raw_h5m_sha256": "a" * 64,
            "canonical_geometry_fingerprint": "b" * 64,
        }
    ]


def _spectrum(particle):
    return {
        "domain_id": "magnet-0000",
        "observable": "scalar_flux_spectrum",
        "particle": particle,
        "estimator": "tracklength",
        "normalization": "per_source_history",
        "unit": "1/cm2/source_history/bin",
        "energy_edges_eV": [0.0, 1.0e6, 2.0e7],
        "mean_per_source": [0.25, 0.5],
        "std_dev_per_source": [0.01, 0.02],
        "tally_id": f"magnet-0000-{particle}-spectrum",
        "statepoint_sha256": "d" * 64,
        "raw_h5m_sha256": "a" * 64,
        "canonical_geometry_fingerprint": "b" * 64,
        "tally_definition_sha256": "0" * 64,
        "model_xml_sha256": "5" * 64,
        "settings_xml_sha256": "6" * 64,
        "strict_run_audit_sha256": "7" * 64,
        "root_acceptance_receipt_sha256": "8" * 64,
        "source_histories": 1000,
        "openmc_cell_id": 9001,
        "dagmc_volume_id": 9,
        "volume_cm3": 12.0,
        "volume_receipt_sha256": "9" * 64,
    }


def _estimators():
    return [
        _spectrum("neutron"),
        _spectrum("photon"),
        {
            "domain_id": "magnet-0000",
            "observable": "damage_energy",
            "particle": "neutron",
            "estimator": "tracklength",
            "normalization": "per_source_history",
            "unit": "eV/source_history",
            "mean_per_source": [42.0],
            "std_dev_per_source": [2.0],
            "tally_id": "magnet-0000-damage-energy",
            "statepoint_sha256": "d" * 64,
            "raw_h5m_sha256": "a" * 64,
            "canonical_geometry_fingerprint": "b" * 64,
            "tally_definition_sha256": "0" * 64,
            "model_xml_sha256": "5" * 64,
            "settings_xml_sha256": "6" * 64,
            "strict_run_audit_sha256": "7" * 64,
            "root_acceptance_receipt_sha256": "8" * 64,
            "source_histories": 1000,
            "openmc_cell_id": 9001,
            "dagmc_volume_id": 9,
            "volume_cm3": 12.0,
            "volume_receipt_sha256": "9" * 64,
        },
        {
            "domain_id": "magnet-0000",
            "observable": "reaction_rate",
            "particle": "neutron",
            "nuclide": "W184",
            "reaction": "(n,gamma)",
            "estimator": "analog",
            "normalization": "per_source_history",
            "unit": "reactions/source_history",
            "mean_per_source": [0.001],
            "std_dev_per_source": [0.0001],
            "tally_id": "magnet-0000-W184-ngamma",
            "statepoint_sha256": "d" * 64,
            "raw_h5m_sha256": "a" * 64,
            "canonical_geometry_fingerprint": "b" * 64,
            "tally_definition_sha256": "0" * 64,
            "model_xml_sha256": "5" * 64,
            "settings_xml_sha256": "6" * 64,
            "strict_run_audit_sha256": "7" * 64,
            "root_acceptance_receipt_sha256": "8" * 64,
            "source_histories": 1000,
            "openmc_cell_id": 9001,
            "dagmc_volume_id": 9,
            "volume_cm3": 12.0,
            "volume_receipt_sha256": "9" * 64,
        },
    ]


def _bank():
    return {
        "schema": "parastell.magnet_boundary_source/v2.2.0",
        "artifact_sha256": "1" * 64,
        "envelope_manifest_sha256": "2" * 64,
        "bank_classification": "COMPLETE_CROSSING_BANK",
        "canonical_weight_semantics": (
            "raw_openmc_weight_divided_only_by_exact_source_histories"
        ),
        "fields": list(PHASE_SPACE_FIELDS),
        "joint_records_preserved": True,
        "parent_history_available": False,
        "polarization_available": False,
        "statepoint_sha256": "d" * 64,
        "raw_h5m_sha256": "a" * 64,
        "canonical_geometry_fingerprint": "b" * 64,
        "strict_run_audit_sha256": "7" * 64,
        "root_acceptance_receipt_sha256": "8" * 64,
    }


def _schedule():
    return build_activation_schedule_reference(
        schedule_id="wistell-d-full-power-checkpoints-v1",
        schedule_sha256="3" * 64,
        verification_receipt_path=(
            Path(__file__).parent
            / "data"
            / "activation_schedule_verification_receipt.json"
        ),
    )


def _bundle():
    return build_radiation_consumer_handoff(
        provenance=_provenance(),
        materials=_materials(),
        volume_estimators=_estimators(),
        boundary_phase_space=_bank(),
        activation_schedule_reference=_schedule(),
    )


def test_bounded_workflow_bundle_is_valid_without_claiming_statistics():
    bundle = _bundle()
    assert bundle["status"] == "WORKFLOW_CONTRACT_VALID"
    assert bundle["provenance"]["statistics_classification"] == (
        "WORKFLOW_SMOKE_ONLY"
    )
    assert (
        bundle["consumer_routes"]["activation"][
            "surface_bank_allowed_as_flux_substitute"
        ]
        is False
    )
    assert (
        bundle["activation_schedule_reference"][
            "production_activation_authorized"
        ]
        is False
    )


def test_spectra_pka_projection_matches_existing_consumer_contract():
    rows = spectra_pka_inputs(_bundle())
    assert len(rows) == 1
    row = rows[0]
    required = {
        "layer_id",
        "material_id",
        "energy_bin_eV",
        "bin_integrated_flux",
        "openmc_version",
        "nuclear_data_library",
        "cross_sections_xml_sha256",
        "tally_id",
        "geometry_hash",
    }
    assert required <= set(row)
    assert row["bin_integrated_flux"] == [0.25, 0.5]
    assert row["normalization"] == "per_source_history"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda bundle: bundle["volume_estimators"][0].update(
                unit="1/cm2/source_history/eV"
            ),
            "bin integrated",
        ),
        (
            lambda bundle: bundle["boundary_phase_space"].update(
                bank_classification="TRUNCATED_INVALID_BANK"
            ),
            "complete crossing bank",
        ),
        (
            lambda bundle: bundle["boundary_phase_space"].update(
                fields=list(PHASE_SPACE_FIELDS[:-1])
            ),
            "missing phase fields",
        ),
        (
            lambda bundle: bundle["boundary_phase_space"].update(
                canonical_weight_semantics="tally_conditioned"
            ),
            "reconditioned",
        ),
        (
            lambda bundle: bundle["consumer_routes"]["activation"].update(
                surface_bank_allowed_as_flux_substitute=True
            ),
            "surface current",
        ),
        (
            lambda bundle: bundle["consumer_routes"][
                "deterministic_transport"
            ].update(projection_may_replace_canonical_bank=True),
            "replaced canonical",
        ),
        (
            lambda bundle: bundle["activation_schedule_reference"].update(
                inline_executable_segments=True
            ),
            "cannot own depletion segments",
        ),
    ],
)
def test_bundle_rejects_consumer_and_normalization_mixing(mutation, match):
    bundle = copy.deepcopy(_bundle())
    mutation(bundle)
    with pytest.raises(RadiationHandoffError, match=match):
        validate_radiation_consumer_handoff(bundle)


def test_every_material_domain_requires_a_neutron_scalar_spectrum():
    bundle = _bundle()
    bundle["materials"].append(
        {
            "domain_id": "magnet-0001",
            "material_id": "homogenized-magnet",
            "composition_sha256": "4" * 64,
            "volume_cm3": 10.0,
            "openmc_cell_id": 9002,
            "dagmc_volume_id": 10,
            "volume_receipt_sha256": "5" * 64,
            "raw_h5m_sha256": "a" * 64,
            "canonical_geometry_fingerprint": "b" * 64,
        }
    )
    with pytest.raises(RadiationHandoffError, match="every material domain"):
        validate_radiation_consumer_handoff(bundle)


def test_downstream_export_adds_material_identity_and_preserves_ownership(
    tmp_path,
):
    bundle = _bundle()
    bundle["materials"][0].update(
        {
            "density_g_cm3": 6.3,
            "temperature_K": 600.0,
            "composition_basis": "atom_fraction",
            "isotopes": {"B10": 0.2, "B11": 0.8},
        }
    )
    exports = build_downstream_exports(
        bundle,
        ownership_contribution_id="wistell-d:magnet-0000:run-1",
        delayed_photon_source_id="wistell-d:magnet-0000:delayed-1",
    )
    assert exports["status"] == "IMPORT_INPUTS_VALIDATED"
    assert exports["spectra_pka"][0]["material"]["isotopes"] == {
        "B10": 0.2,
        "B11": 0.8,
    }
    assert (
        exports["magnet_boundary_replay"][
            "source_rate_may_be_applied_more_than_once"
        ]
        is False
    )
    paths = write_downstream_exports(tmp_path / "exports", exports)
    assert len(paths) == 5
    assert all(path.is_file() for path in paths)


def test_downstream_export_rejects_prompt_delayed_ownership_collision():
    bundle = _bundle()
    bundle["materials"][0].update(
        {
            "density_g_cm3": 6.3,
            "temperature_K": 600.0,
            "composition_basis": "atom_fraction",
            "isotopes": {"B10": 0.2, "B11": 0.8},
        }
    )
    with pytest.raises(ValueError, match="collide"):
        build_downstream_exports(
            bundle,
            ownership_contribution_id="run-1",
            delayed_photon_source_id="run-1:prompt-boundary",
        )


def test_reaction_rate_requires_nuclide_and_reaction_identity():
    bundle = _bundle()
    bundle["volume_estimators"][-1]["nuclide"] = ""
    with pytest.raises(RadiationHandoffError, match="identity is incomplete"):
        validate_radiation_consumer_handoff(bundle)


def test_schedule_reference_cannot_copy_or_change_source_rate():
    reference = _schedule()
    reference["full_power_rate_binding"]["copied_rate_per_s"] = 2.7e20
    with pytest.raises(RadiationHandoffError, match="source-rate binding"):
        build_radiation_consumer_handoff(
            provenance=_provenance(),
            materials=_materials(),
            volume_estimators=_estimators(),
            boundary_phase_space=_bank(),
            activation_schedule_reference=reference,
        )
