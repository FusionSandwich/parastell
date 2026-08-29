from __future__ import annotations

import copy
from pathlib import Path

import h5py
import numpy as np
import pytest

from parastell.openmc16_response_handoff import (
    build_exact_tally_bindings,
    build_openmc16_radiation_consumer_handoff,
)
from parastell.openmc16_response_results import read_openmc16_response_set
from parastell.radiation_consumer_handoff import (
    PHASE_SPACE_FIELDS,
    RadiationHandoffError,
    build_activation_schedule_reference,
    validate_radiation_consumer_handoff,
)

TALLY_NAMES = ("flux", "heating", "damage", "gas", "reaction", "production")


def _write_tally(tallies, *, index, name, filters, scores, nuclides, means):
    tally = tallies.create_group(f"tally {index}")
    tally["name"] = np.bytes_(name)
    tally["filters"] = filters
    tally["score_bins"] = [np.bytes_(value) for value in scores]
    tally["nuclides"] = [np.bytes_(value) for value in nuclides]
    tally["estimator"] = np.bytes_("tracklength")
    tally["n_realizations"] = 3
    means = np.asarray(means, dtype=float).reshape(-1)
    deviations = np.full_like(means, 0.01)
    sums = 3 * means
    squares = 3 * (means**2 + deviations**2 * 2)
    responses = len(scores) * len(nuclides)
    tally["results"] = np.stack([sums, squares], axis=-1).reshape(
        -1, responses, 2
    )


def _statepoint(path: Path, *, malformed_production=False):
    with h5py.File(path, "w") as handle:
        handle.attrs["filetype"] = np.bytes_("statepoint")
        handle.attrs["openmc_version"] = [0, 16, 0]
        handle["run_mode"] = np.bytes_("fixed source")
        handle["n_particles"] = 10
        handle["n_batches"] = 3
        handle["current_batch"] = 3
        handle["seed"] = 9123
        tallies = handle.create_group("tallies")
        filters = tallies.create_group("filters")

        cell = filters.create_group("filter 1")
        cell["type"] = np.bytes_("cell")
        cell["n_bins"] = 1
        cell["bins"] = [9008]
        particle = filters.create_group("filter 2")
        particle["type"] = np.bytes_("particle")
        particle["n_bins"] = 1
        particle["bins"] = [np.bytes_("neutron")]
        energy = filters.create_group("filter 3")
        energy["type"] = np.bytes_("energy")
        energy["n_bins"] = 2
        energy["bins"] = [0.0, 1.0e6, 2.0e7]
        reaction = filters.create_group("filter 4")
        reaction["type"] = np.bytes_("reaction")
        reaction["n_bins"] = 2
        reaction["bins"] = [np.bytes_("(n,elastic)"), np.bytes_("(n,gamma)")]
        production = filters.create_group("filter 5")
        production["type"] = np.bytes_("particleproduction")
        production["n_bins"] = 3 if malformed_production else 2
        production["particles"] = [np.bytes_("photon")]
        production["energies"] = [0.0, 1.0e6, 2.0e7]

        _write_tally(
            tallies,
            index=20,
            name="flux",
            filters=[1, 2, 3],
            scores=["flux"],
            nuclides=["total"],
            means=[12.0, 24.0],
        )
        _write_tally(
            tallies,
            index=21,
            name="heating",
            filters=[1, 2, 3],
            scores=["heating"],
            nuclides=["total"],
            means=[2.0, 3.0],
        )
        _write_tally(
            tallies,
            index=22,
            name="damage",
            filters=[1, 2, 3],
            scores=["damage-energy"],
            nuclides=["total"],
            means=[4.0, 5.0],
        )
        _write_tally(
            tallies,
            index=23,
            name="gas",
            filters=[1, 2, 3],
            scores=[
                "H1-production",
                "H2-production",
                "H3-production",
                "He3-production",
                "He4-production",
            ],
            nuclides=["total"],
            means=np.arange(1.0, 11.0),
        )
        _write_tally(
            tallies,
            index=24,
            name="reaction",
            filters=[1, 2, 3, 4],
            scores=["events"],
            nuclides=["W184"],
            means=[0.1, 0.2, 0.3, 0.4],
        )
        _write_tally(
            tallies,
            index=25,
            name="production",
            filters=[1, 2, 3, 5],
            scores=["events"],
            nuclides=["total"],
            means=(
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
                if malformed_production
                else [0.01, 0.02, 0.03, 0.04]
            ),
        )


def _bindings():
    common = {
        "filter_types": ["cell", "particle", "energy"],
        "nuclides": ["total"],
    }
    return {
        "flux": {
            **common,
            "scores": ["flux"],
            "tally_definition_sha256": "1" * 64,
        },
        "heating": {
            **common,
            "scores": ["heating"],
            "tally_definition_sha256": "2" * 64,
        },
        "damage": {
            **common,
            "scores": ["damage-energy"],
            "tally_definition_sha256": "3" * 64,
        },
        "gas": {
            **common,
            "scores": [
                "H1-production",
                "H2-production",
                "H3-production",
                "He3-production",
                "He4-production",
            ],
            "tally_definition_sha256": "4" * 64,
        },
        "reaction": {
            "filter_types": ["cell", "particle", "energy", "reaction"],
            "scores": ["events"],
            "nuclides": ["W184"],
            "reaction_mt_by_bin": {"(n,elastic)": 2, "(n,gamma)": 102},
            "tally_definition_sha256": "5" * 64,
        },
        "production": {
            "filter_types": [
                "cell",
                "particle",
                "energy",
                "particleproduction",
            ],
            "scores": ["events"],
            "nuclides": ["total"],
            "tally_definition_sha256": "6" * 64,
        },
    }


def _provenance(statepoint_sha256):
    return {
        "raw_h5m_sha256": "a" * 64,
        "canonical_geometry_fingerprint": "b" * 64,
        "source_definition_sha256": "c" * 64,
        "statepoint_sha256": statepoint_sha256,
        "nuclear_data_sha256": "e" * 64,
        "model_xml_sha256": "5" * 64,
        "settings_xml_sha256": "6" * 64,
        "strict_run_audit_sha256": "7" * 64,
        "root_acceptance_receipt_sha256": "8" * 64,
        "nuclear_data_library": "ENDF/B-VIII.0",
        "openmc_version": "0.16.0",
        "source_histories": 30,
        "physical_source_rate_per_s": 1.0e20,
        "source_rate_scope": "modeled_period",
        "normalization": "per_source_history",
        "statistics_classification": "WORKFLOW_SMOKE_ONLY",
    }


def _materials():
    return [
        {
            "domain_id": "magnet-0008",
            "material_id": "homogenized-magnet",
            "composition_sha256": "f" * 64,
            "volume_cm3": 12.0,
            "openmc_cell_id": 9008,
            "dagmc_volume_id": 17,
            "volume_receipt_sha256": "9" * 64,
            "raw_h5m_sha256": "a" * 64,
            "canonical_geometry_fingerprint": "b" * 64,
        }
    ]


def _bank(statepoint_sha256):
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
        "statepoint_sha256": statepoint_sha256,
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


def _bundle(tmp_path):
    path = tmp_path / "statepoint.3.h5"
    _statepoint(path)
    response_set = read_openmc16_response_set(path, TALLY_NAMES)
    provenance = _provenance(response_set["statepoint_sha256"])
    bundle = build_openmc16_radiation_consumer_handoff(
        response_set=response_set,
        tally_bindings=_bindings(),
        provenance=provenance,
        materials=_materials(),
        boundary_phase_space=_bank(response_set["statepoint_sha256"]),
        activation_schedule_reference=_schedule(),
    )
    return bundle, response_set


def test_statepoint_results_reach_existing_consumer_validator(tmp_path):
    bundle, _ = _bundle(tmp_path)
    validate_radiation_consumer_handoff(bundle)
    assert bundle["producer_adapter"]["integrated_totals_derived"] is False
    assert bundle["producer_adapter"]["covariance_status"] == (
        "UNAVAILABLE_NOT_FABRICATED"
    )
    rows = bundle["volume_estimators"]
    neutron_flux = [
        row for row in rows if row["observable"] == "scalar_flux_spectrum"
    ]
    assert neutron_flux[0]["mean_per_source"] == [1.0, 2.0]
    damage = next(row for row in rows if row["observable"] == "damage_energy")
    assert damage["unit"] == "eV/source_history/bin"
    assert "bins are not summed" in damage["integration_semantics"]
    gas = {row["gas_isotope"] for row in rows if "gas_isotope" in row}
    assert gas == {"H1", "H2", "H3", "He3", "He4"}
    reaction = [row for row in rows if row["observable"] == "reaction_rate"]
    assert {(row["mt"], row["reaction_filter_bin"]) for row in reaction} == {
        (2, "(n,elastic)"),
        (102, "(n,gamma)"),
    }
    produced = [
        row for row in rows if row["observable"] == "particle_production"
    ]
    assert {tuple(row["outgoing_energy_bin_eV"]) for row in produced} == {
        (0.0, 1.0e6),
        (1.0e6, 2.0e7),
    }
    assert {row["produced_particle"] for row in produced} == {"photon"}


def test_exact_bindings_capture_axes_and_require_reaction_mt_control(tmp_path):
    path = tmp_path / "statepoint.3.h5"
    _statepoint(path)
    response_set = read_openmc16_response_set(path, TALLY_NAMES)
    bindings = build_exact_tally_bindings(
        response_set,
        model_xml_sha256="a" * 64,
        reaction_mt_by_tally={
            "reaction": {"(n,elastic)": 2, "(n,gamma)": 102}
        },
    )
    assert bindings["status"] == "PASS"
    assert set(bindings["tallies"]) == set(TALLY_NAMES)
    assert bindings["tallies"]["reaction"]["reaction_mt_by_bin"] == {
        "(n,elastic)": 2,
        "(n,gamma)": 102,
    }
    assert len(bindings["tallies"]["flux"]["tally_definition_sha256"]) == 64
    with pytest.raises(ValueError, match="ReactionFilter MT control"):
        build_exact_tally_bindings(
            response_set,
            model_xml_sha256="a" * 64,
            reaction_mt_by_tally={"reaction": {"(n,elastic)": 2}},
        )


def test_adapter_rejects_missing_reaction_mt_identity(tmp_path):
    path = tmp_path / "statepoint.3.h5"
    _statepoint(path)
    response_set = read_openmc16_response_set(path, TALLY_NAMES)
    bindings = _bindings()
    del bindings["reaction"]["reaction_mt_by_bin"]["(n,gamma)"]
    with pytest.raises(ValueError, match="MT identities"):
        build_openmc16_radiation_consumer_handoff(
            response_set=response_set,
            tally_bindings=bindings,
            provenance=_provenance(response_set["statepoint_sha256"]),
            materials=_materials(),
            boundary_phase_space=_bank(response_set["statepoint_sha256"]),
            activation_schedule_reference=_schedule(),
        )


def test_reader_rejects_malformed_particleproduction_identity(tmp_path):
    path = tmp_path / "statepoint.3.h5"
    _statepoint(path, malformed_production=True)
    with pytest.raises(ValueError, match="bin count is inconsistent"):
        read_openmc16_response_set(path, ["production"])


def test_adapter_rejects_ambiguous_or_unbound_axes(tmp_path):
    path = tmp_path / "statepoint.3.h5"
    _statepoint(path)
    response_set = read_openmc16_response_set(path, TALLY_NAMES)
    response_set["responses"][0]["filters"][2]["type"] = "mystery"
    bindings = _bindings()
    bindings["flux"]["filter_types"][2] = "mystery"
    with pytest.raises(ValueError, match="ambiguous response axis"):
        build_openmc16_radiation_consumer_handoff(
            response_set=response_set,
            tally_bindings=bindings,
            provenance=_provenance(response_set["statepoint_sha256"]),
            materials=_materials(),
            boundary_phase_space=_bank(response_set["statepoint_sha256"]),
            activation_schedule_reference=_schedule(),
        )


def test_consumer_rejects_unit_and_hash_tampering(tmp_path):
    bundle, _ = _bundle(tmp_path)
    unit_tamper = copy.deepcopy(bundle)
    next(
        row
        for row in unit_tamper["volume_estimators"]
        if row["observable"] == "heating"
    )["unit"] = "eV/source_history"
    with pytest.raises(RadiationHandoffError, match="unit is invalid"):
        validate_radiation_consumer_handoff(unit_tamper)
    hash_tamper = copy.deepcopy(bundle)
    hash_tamper["volume_estimators"][0]["mean_per_source"][0] += 1.0
    with pytest.raises(RadiationHandoffError, match="payload was modified"):
        validate_radiation_consumer_handoff(hash_tamper)
