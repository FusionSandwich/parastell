from __future__ import annotations

import copy
from pathlib import Path

import pytest

from parastell.activation_handoff import (
    ActivationHandoffError,
    DIRECT90_REFERENCE_MANIFEST_SHA256,
    ENDFB80_FAST_CHAIN_SHA256,
    FULL_TRANSPORT_CATALOG_SHA256,
    FULL_TRANSPORT_PAYLOAD_LEDGER_SHA256,
    DIRECT90_MODELED_SOURCE_RATE_PER_S,
    DIRECT90_SOURCE_CANONICAL_FINGERPRINT,
    DIRECT90_SOURCE_EXPANSION_MANIFEST_SHA256,
    DIRECT90_SOURCE_MESH_SHA256,
    DIRECT90_STRENGTHS_SHA256,
    HALF45_SOURCE_MESH_SHA256,
    HALF45_STRENGTHS_SHA256,
    build_activation_handoff,
    inspect_activation_data,
    validate_activation_handoff,
)
from parastell.radiation_consumer_handoff import (
    build_activation_schedule_reference,
)


def _geometry(native="PASS", navigation="PASS"):
    return {
        "reference_manifest_sha256": DIRECT90_REFERENCE_MANIFEST_SHA256,
        "modeled_toroidal_extent_degrees": 90.0,
        "raw_h5m_sha256": "a" * 64,
        "canonical_geometry_fingerprint": "b" * 64,
        "native_volume_audit_receipt_sha256": "6" * 64,
        "magnet_representation": "continuous_30_cm_magnet_envelope",
        "native_geometry_gate": native,
        "openmc_navigation_gate": navigation,
        "transport_periodic_wrapper": {
            "outside_h5m": True,
            "periodic_planes_paired": True,
            "n_field_periods": 4,
            "extent_degrees": 90.0,
        },
    }


def _volumes():
    materials = [
        "Vacuum",
        "first_wall",
        "breeder",
        "back_wall",
        "high_temperature_shield",
        "vacuum_vessel",
        "low_temperature_shield",
        "Vacuum",
        "magnet_envelope",
    ]
    return [
        {
            "volume_id": index,
            "material_tag": material,
            "volume_cm3": 10.0,
            "volume_audit_receipt_sha256": "6" * 64,
        }
        for index, material in enumerate(materials, start=1)
    ]


def _data():
    return {
        "chain": {"sha256": ENDFB80_FAST_CHAIN_SHA256, "qualified": True},
        "transport_catalog": {
            "sha256": FULL_TRANSPORT_CATALOG_SHA256,
            "qualified": True,
            "payload_count": 728,
            "present_payload_count": 728,
            "payloads_all_present": True,
            "payloads_all_hashed": True,
            "payload_ledger_sha256": FULL_TRANSPORT_PAYLOAD_LEDGER_SHA256,
        },
    }


def _source_mesh():
    return {
        "sha256": DIRECT90_SOURCE_MESH_SHA256,
        "strengths_sha256": DIRECT90_STRENGTHS_SHA256,
        "expansion_manifest_sha256": DIRECT90_SOURCE_EXPANSION_MANIFEST_SHA256,
        "parent_half_source_mesh_sha256": HALF45_SOURCE_MESH_SHA256,
        "parent_half_strengths_sha256": HALF45_STRENGTHS_SHA256,
        "canonical_source_fingerprint": DIRECT90_SOURCE_CANONICAL_FINGERPRINT,
        "geometry_fingerprint": "b" * 64,
        "status": "PASS",
        "modeled_toroidal_extent_degrees": 90.0,
        "source_rate_scope": "modeled_90_degree_period",
        "strength_sum_per_s": DIRECT90_MODELED_SOURCE_RATE_PER_S,
        "domain_element_order_audit_status": "PASS",
        "audit_receipt_sha256": "7" * 64,
        "accepted_geometry_containment_run": True,
        "containment_status": "PASS",
        "containment_receipt_sha256": "8" * 64,
        "containment_raw_h5m_sha256": "a" * 64,
    }


def _domain():
    return {
        "domain_id": "whole-magnet-envelope",
        "domain_kind": "WHOLE_MAGNET_ENVELOPE",
        "dagmc_volume_id": 9,
        "openmc_cell_id": 9009,
        "cell_id_mapping_method": "openmc_geometry_introspection",
        "dagmc_id_inference_used": False,
        "material_tag": "magnet_envelope",
        "openmc_material_id": 109,
        "volume_cm3": 10.0,
        "density_g_cm3": 6.0,
        "isotopic_composition_sha256": "f" * 64,
        "id_mapping_receipt_sha256": "8" * 64,
        "raw_h5m_sha256": "a" * 64,
        "geometry_fingerprint": "b" * 64,
        "boundary_surface_ids": [101, 102, 103, 104],
        "boundary_closure": "EUCLIDEAN_CLOSED",
    }


def _flux():
    rate = DIRECT90_MODELED_SOURCE_RATE_PER_S
    return {
        "status": "PASS",
        "path": "scalar_flux_fields.h5",
        "sha256": "c" * 64,
        "statepoint_sha256": "1" * 64,
        "tally_definition_sha256": "2" * 64,
        "raw_h5m_sha256": "a" * 64,
        "geometry_fingerprint": "b" * 64,
        "source_mesh_sha256": DIRECT90_SOURCE_MESH_SHA256,
        "openmc_version": "0.16.0",
        "particle": "neutron",
        "score": "flux",
        "estimator": "tracklength",
        "filters": ["cell", "energy"],
        "source_histories": 500000,
        "physical_source_rate_per_s": rate,
        "domains": [
            {
                "domain_id": "whole-magnet-envelope",
                "volume_cm3": 10.0,
                "energy_edges_eV": [0.0, 1.0e6, 2.0e7],
                "track_length_mean_cm_per_source": [2.0, 3.0],
                "track_length_std_dev_cm_per_source": [0.1, 0.2],
                "physical_scalar_flux_per_cm2_s": [
                    2.0 / 10.0 * rate,
                    3.0 / 10.0 * rate,
                ],
            }
        ],
    }


def _handoff(
    *,
    scalar_flux=None,
    native="PASS",
    navigation="PASS",
    bind_domain=False,
    bind_schedule=True,
):
    return build_activation_handoff(
        geometry=_geometry(native=native, navigation=navigation),
        volumes=_volumes(),
        scalar_flux=scalar_flux,
        activation_data=_data(),
        physical_source_rate_per_s=DIRECT90_MODELED_SOURCE_RATE_PER_S,
        source_mesh=_source_mesh() if bind_domain else None,
        activation_domains=[_domain()] if bind_domain else (),
        activation_schedule_reference=(
            build_activation_schedule_reference(
                schedule_id="wistell-d-full-power-checkpoints-v1",
                schedule_sha256="3" * 64,
                verification_receipt_path=(
                    Path(__file__).parent
                    / "data"
                    / "activation_schedule_verification_receipt.json"
                ),
            )
            if bind_domain and bind_schedule
            else None
        ),
    )


def test_activation_handoff_is_blocked_until_medium_scalar_flux_exists():
    handoff = _handoff()
    assert handoff["status"] == (
        "BLOCKED_PENDING_ACCEPTED_GEOMETRY_AND_MEDIUM_FIELD"
    )
    assert (
        handoff["activation_input"]["boundary_bank_used_for_activation"]
        is False
    )
    assert handoff["old_geometry_activation_results"]["status"] == (
        "REFERENCE_ONLY_REJECTED_GEOMETRY"
    )


def test_activation_handoff_becomes_ready_only_after_both_geometry_gates():
    flux = _flux()
    assert _handoff(scalar_flux=flux, bind_domain=True)["status"] == (
        "READY_FOR_DPA_ACTIVATION_QUALIFICATION"
    )
    assert _handoff(scalar_flux=flux, native="FAIL", bind_domain=True)[
        "status"
    ].startswith("BLOCKED_")
    assert _handoff(scalar_flux=flux, navigation="NOT_RUN", bind_domain=True)[
        "status"
    ].startswith("BLOCKED_")


def test_activation_handoff_stays_blocked_without_dpa_schedule_reference():
    handoff = _handoff(
        scalar_flux=_flux(), bind_domain=True, bind_schedule=False
    )
    assert handoff["status"].startswith("BLOCKED_")


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (
            ("activation_input", "observable"),
            "surface_current",
            "volume scalar flux",
        ),
        (
            ("activation_input", "boundary_bank_used_for_activation"),
            True,
            "boundary current",
        ),
        (("geometry", "modeled_toroidal_extent_degrees"), 45.0, "90-degree"),
        (
            ("geometry", "magnet_representation"),
            "split_casing_winding",
            "representation",
        ),
        (("activation_data", "chain", "sha256"), "d" * 64, "chain hash"),
    ],
)
def test_activation_handoff_rejects_unsafe_or_wrong_provenance(
    path, value, match
):
    handoff = _handoff()
    mutated = copy.deepcopy(handoff)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ActivationHandoffError, match=match):
        validate_activation_handoff(mutated)


def test_activation_handoff_rejects_old_selected_six_material_catalog():
    handoff = _handoff()
    handoff["activation_data"]["transport_catalog"][
        "sha256"
    ] = "ad0edc324ead933f13f4931af15670403234ff4337e4da95a671ad0ba46aade3"
    with pytest.raises(ActivationHandoffError, match="transport_catalog hash"):
        validate_activation_handoff(handoff)


def test_activation_handoff_rejects_duplicate_volume_ids():
    handoff = _handoff()
    handoff["volume_inventory"][1]["volume_id"] = 1
    with pytest.raises(ActivationHandoffError, match="unique positive"):
        validate_activation_handoff(handoff)


def test_ready_handoff_rejects_dagmc_to_openmc_cell_id_inference():
    handoff = _handoff(scalar_flux=_flux(), bind_domain=True)
    handoff["activation_domains"][0]["dagmc_id_inference_used"] = True
    with pytest.raises(ActivationHandoffError, match="cannot be inferred"):
        validate_activation_handoff(handoff)


def test_domain_material_must_match_native_volume_inventory():
    handoff = _handoff(scalar_flux=_flux(), bind_domain=True)
    handoff["activation_domains"][0]["dagmc_volume_id"] = 1
    with pytest.raises(ActivationHandoffError, match="disagrees"):
        validate_activation_handoff(handoff)


def test_ready_handoff_rejects_scalar_flux_normalization_drift():
    handoff = _handoff(scalar_flux=_flux(), bind_domain=True)
    handoff["activation_input"]["scalar_flux"]["domains"][0][
        "physical_scalar_flux_per_cm2_s"
    ][0] *= 1.01
    with pytest.raises(ActivationHandoffError, match="normalization closure"):
        validate_activation_handoff(handoff)


def test_scalar_flux_denominator_must_match_audited_domain_volume():
    handoff = _handoff(scalar_flux=_flux(), bind_domain=True)
    handoff["activation_input"]["scalar_flux"]["domains"][0][
        "volume_cm3"
    ] = 11.0
    with pytest.raises(ActivationHandoffError, match="disagrees"):
        validate_activation_handoff(handoff)


def test_domain_volume_must_match_native_audited_volume():
    handoff = _handoff(scalar_flux=_flux(), bind_domain=True)
    handoff["activation_domains"][0]["volume_cm3"] = 11.0
    handoff["activation_input"]["scalar_flux"]["domains"][0][
        "volume_cm3"
    ] = 11.0
    with pytest.raises(ActivationHandoffError, match="native audited"):
        validate_activation_handoff(handoff)


def test_duplicate_scalar_domain_rows_are_rejected():
    handoff = _handoff(scalar_flux=_flux(), bind_domain=True)
    handoff["activation_input"]["scalar_flux"]["domains"].append(
        copy.deepcopy(handoff["activation_input"]["scalar_flux"]["domains"][0])
    )
    with pytest.raises(ActivationHandoffError, match="domains do not match"):
        validate_activation_handoff(handoff)


def test_scalar_flux_is_validated_even_when_navigation_gate_is_blocked():
    handoff = _handoff(
        scalar_flux=_flux(), navigation="NOT_RUN", bind_domain=True
    )
    handoff["activation_input"]["scalar_flux"]["openmc_version"] = "0.15.0"
    with pytest.raises(ActivationHandoffError, match="OpenMC 0.16.0"):
        validate_activation_handoff(handoff)


def test_source_rate_must_close_to_authoritative_90_degree_mesh():
    with pytest.raises(ActivationHandoffError, match="close to source mesh"):
        build_activation_handoff(
            geometry=_geometry(),
            volumes=_volumes(),
            scalar_flux=None,
            activation_data=_data(),
            physical_source_rate_per_s=2.693734274881251e20,
            source_mesh=_source_mesh(),
            activation_domains=[_domain()],
        )


def test_transport_catalog_requires_payload_hash_ledger():
    handoff = _handoff()
    handoff["activation_data"]["transport_catalog"][
        "payloads_all_hashed"
    ] = False
    with pytest.raises(ActivationHandoffError, match="payload hash ledger"):
        validate_activation_handoff(handoff)


def test_source_mesh_requires_domain_and_element_order_audit_receipt():
    source = _source_mesh()
    source["domain_element_order_audit_status"] = "PENDING"
    with pytest.raises(ActivationHandoffError, match="has not passed"):
        build_activation_handoff(
            geometry=_geometry(),
            volumes=_volumes(),
            scalar_flux=None,
            activation_data=_data(),
            physical_source_rate_per_s=DIRECT90_MODELED_SOURCE_RATE_PER_S,
            source_mesh=source,
            activation_domains=[_domain()],
        )


def test_half_period_source_is_rejected_as_direct90_even_with_forged_extent():
    source = _source_mesh()
    source["sha256"] = HALF45_SOURCE_MESH_SHA256
    source["strengths_sha256"] = HALF45_STRENGTHS_SHA256
    with pytest.raises(ActivationHandoffError, match="source mesh hash"):
        build_activation_handoff(
            geometry=_geometry(),
            volumes=_volumes(),
            scalar_flux=None,
            activation_data=_data(),
            physical_source_rate_per_s=DIRECT90_MODELED_SOURCE_RATE_PER_S,
            source_mesh=source,
            activation_domains=[_domain()],
        )


def test_source_containment_pending_cannot_unlock_activation():
    source = _source_mesh()
    source["accepted_geometry_containment_run"] = False
    source["containment_status"] = "PENDING"
    with pytest.raises(ActivationHandoffError, match="containment"):
        build_activation_handoff(
            geometry=_geometry(),
            volumes=_volumes(),
            scalar_flux=None,
            activation_data=_data(),
            physical_source_rate_per_s=DIRECT90_MODELED_SOURCE_RATE_PER_S,
            source_mesh=source,
            activation_domains=[_domain()],
        )


def test_activation_data_inspection_hashes_every_catalog_payload(
    tmp_path, monkeypatch
):
    chain = tmp_path / "chain.xml"
    chain.write_text("<depletion_chain/>", encoding="utf-8")
    payload = tmp_path / "H1.h5"
    payload.write_bytes(b"payload")
    catalog = tmp_path / "cross_sections.xml"
    catalog.write_text(
        '<cross_sections><library materials="H1" path="H1.h5" '
        'type="neutron"/></cross_sections>',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "parastell.activation_handoff.ENDFB80_FAST_CHAIN_SHA256",
        __import__("hashlib").sha256(chain.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "parastell.activation_handoff.FULL_TRANSPORT_CATALOG_SHA256",
        __import__("hashlib").sha256(catalog.read_bytes()).hexdigest(),
    )
    data = inspect_activation_data(
        chain_path=chain,
        cross_sections_path=catalog,
        hash_transport_payloads=True,
    )
    row = data["transport_catalog"]
    assert row["payload_count"] == 1
    assert row["present_payload_count"] == 1
    assert row["payloads_all_present"] is True
    assert row["payloads_all_hashed"] is True
    assert len(row["payload_ledger_sha256"]) == 64


def test_periodic_wrapper_cannot_substitute_for_closed_dagmc_domain():
    handoff = _handoff(scalar_flux=_flux(), bind_domain=True)
    handoff["activation_domains"][0][
        "boundary_closure"
    ] = "CLOSED_MODULO_PERIODIC_PHI_PAIR"
    with pytest.raises(ActivationHandoffError, match="Euclidean-closed"):
        validate_activation_handoff(handoff)
