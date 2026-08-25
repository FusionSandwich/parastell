import hashlib
import json
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from parastell.energy_groups import get_structure
from parastell.magnet_boundary_envelope import (
    EnvelopeSurface,
    MagnetBoundaryEnvelope,
    build_correlated_bank,
    classify_crossing_bank,
    write_handoff,
)
from parastell.magnet_damage_gas import (
    DAMAGE_TALLY,
    GAS_OPENMC_SCORES,
    GAS_PRODUCTION_SCORES,
    GAS_TALLY,
    export_magnet_damage_gas,
)
from parastell.magnet_radiation_field_bundle import (
    _validate_product,
    read_radiation_field_bundle,
    write_radiation_field_bundle,
)
from parastell.magnet_volume_flux import export_scalar_flux_fields


def _surface(sid, role, centroid, normal, toroidal, poloidal, edges):
    return EnvelopeSurface(
        sid,
        role,
        4.0,
        centroid,
        normal,
        toroidal,
        poloidal,
        (-1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        topology_edge_ids=edges,
    )


def _box_envelope():
    surfaces = (
        _surface(
            1,
            "plasma_facing",
            (1, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            ("a", "b", "c", "d"),
        ),
        _surface(
            2,
            "outer",
            (-1, 0, 0),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, 1),
            ("e", "f", "g", "h"),
        ),
        _surface(
            3,
            "toroidal_side",
            (0, 1, 0),
            (0, 1, 0),
            (-1, 0, 0),
            (0, 0, 1),
            ("a", "e", "i", "j"),
        ),
        _surface(
            4,
            "toroidal_side",
            (0, -1, 0),
            (0, -1, 0),
            (1, 0, 0),
            (0, 0, 1),
            ("b", "f", "k", "l"),
        ),
        _surface(
            5,
            "poloidal_side",
            (0, 0, 1),
            (0, 0, 1),
            (1, 0, 0),
            (0, 1, 0),
            ("c", "g", "i", "k"),
        ),
        _surface(
            6,
            "poloidal_side",
            (0, 0, -1),
            (0, 0, -1),
            (-1, 0, 0),
            (0, 1, 0),
            ("d", "h", "j", "l"),
        ),
    )
    return MagnetBoundaryEnvelope(
        "winding-pack-magnet-A", "magnet-A", 88, surfaces, "1" * 64
    )


def _filter(name, bins):
    cls = type(name, (), {})
    result = cls()
    result.bins = bins
    result.num_bins = len(bins)
    return result


def _damage_gas_statepoint():
    filters = [
        _filter("CellFilter", [88]),
        _filter("ParticleFilter", ["neutron"]),
        _filter("EnergyFilter", [(0.0, 20.0e6)]),
    ]

    def tally(name, scores):
        values = np.linspace(0.1, 0.1 * len(scores), len(scores))
        return SimpleNamespace(
            name=name,
            scores=list(scores),
            filters=filters,
            num_nuclides=1,
            num_scores=len(scores),
            mean=values,
            std_dev=values / 10.0,
        )

    tallies = {
        DAMAGE_TALLY: tally(DAMAGE_TALLY, ["damage-energy"]),
        GAS_TALLY: tally(
            GAS_TALLY,
            [GAS_OPENMC_SCORES[name] for name in GAS_PRODUCTION_SCORES],
        ),
    }
    return SimpleNamespace(get_tally=lambda *, name: tallies[name])


def _damage_gas_inventory():
    return {
        "damage_energy": DAMAGE_TALLY,
        "gas_production": GAS_TALLY,
        "response_availability": {
            name: {"available": True, "status": "AVAILABLE"}
            for name in ("damage-energy", *GAS_PRODUCTION_SCORES)
        },
    }


@pytest.fixture
def bundle_inputs(tmp_path):
    provenance = {
        "parastell_commit": "a" * 40,
        "openmc_version": "0.16.0",
        "openmc_commit": "b" * 40,
        "statepoint_sha256": "c" * 64,
        "vmec_sha256": "d" * 64,
        "coils_sha256": "e" * 64,
        "source_mesh_sha256": "f" * 64,
        "histories": 10,
        "batches": 2,
        "seeds": [17],
    }
    geometry = {
        "raw_h5m_sha256": "1" * 64,
        "canonical_geometry_fingerprint": "2" * 64,
    }
    source = {
        "physical_source_rate_per_s": 3.0,
        "source_definition_sha256": "3" * 64,
        "source_mesh_sha256": provenance["source_mesh_sha256"],
    }
    nuclear_data = {"manifest_sha256": "4" * 64}
    materials = {
        "resolved_manifest_sha256": "5" * 64,
        "file_sha256": "6" * 64,
    }

    edges = np.asarray(
        get_structure("CCFE-709", particle="neutron").edges_eV, dtype=float
    )
    track_length = np.linspace(0.01, 0.02, len(edges) - 1)[None, :]
    scalar = tmp_path / "scalar.h5"
    export_scalar_flux_fields(
        scalar,
        fields=[
            {
                "name": "neutron_ccfe_709",
                "field_kind": "whole_volume",
                "particle": "neutron",
                "energy_edges_eV": edges,
                "track_length_mean_cm_per_source": track_length,
                "track_length_std_dev_cm_per_source": track_length / 10.0,
                "volume_cm3": [2.0],
                "region_ids": ["88"],
                "magnet_ids": ["magnet-A"],
                "global_centroid_cm": [[0.0, 0.0, 0.0]],
                "local_centreline_coordinates_cm": [[0.0, 0.0, 0.0]],
                "nearest_centreline_global_cm": [[0.0, 0.0, 0.0]],
                "centreline_arclength_cm": [0.0],
                "normalized_arclength": [0.0],
                "centreline_tangent": [[1.0, 0.0, 0.0]],
                "centreline_radial": [[0.0, 1.0, 0.0]],
                "centreline_transverse": [[0.0, 0.0, 1.0]],
                "distance_to_centreline_cm": [0.0],
                "centreline_linkage_status": [
                    "LINKED_NEAREST_CENTRELINE_SEGMENT"
                ],
                "frame_type": ["coil_centerline_parallel_transport"],
                "frame_quality_status": ["PASS"],
                "material_ids": ["YBCO"],
                "energy_structure": "CCFE-709",
                "resolution_status": np.full((1, 709), 2, dtype=np.uint8),
            }
        ],
        physical_source_rate_per_s=source["physical_source_rate_per_s"],
        provenance={
            "parastell_commit": provenance["parastell_commit"],
            "openmc_commit": provenance["openmc_commit"],
            "vmec_sha256": provenance["vmec_sha256"],
            "coils_sha256": provenance["coils_sha256"],
            "histories": provenance["histories"],
            "batches": provenance["batches"],
            "seeds": provenance["seeds"],
            "raw_h5m_sha256": geometry["raw_h5m_sha256"],
            "canonical_geometry_fingerprint": geometry[
                "canonical_geometry_fingerprint"
            ],
            "source_definition_sha256": source["source_definition_sha256"],
            "source_mesh_sha256": source["source_mesh_sha256"],
            "nuclear_data_manifest_sha256": nuclear_data["manifest_sha256"],
            "statepoint_sha256": provenance["statepoint_sha256"],
            "openmc_version": provenance["openmc_version"],
        },
        material_manifest_sha256=materials["file_sha256"],
    )

    envelope = _box_envelope()
    bank = build_correlated_bank(
        envelope,
        position_global_cm=np.array([[1.0, 0.0, 0.0]]),
        direction_global=np.array([[-1.0, 0.0, 0.0]]),
        energy_eV=[14.1e6],
        raw_weight=[0.1],
        particle=["neutron"],
        surface_id=[1],
        energy_edges_by_particle={
            "neutron": [0.0, 20.0e6],
            "photon": [0.0, 20.0e6],
        },
    )
    bank.metadata["surface_bank_completeness"] = classify_crossing_bank(
        stored_record_count=1,
        selected_record_count=1,
        max_particles_per_file=100,
        max_source_files=1,
        source_file_count=1,
        mpi_ranks=1,
    )
    bank.metadata["zero_record_interpretation"] = {
        "status": "OBSERVED",
        "physical_zero_claimed": False,
        "confidence_level": 0.95,
        "poisson_upper_bound_expected_crossings_per_source": None,
    }
    boundary = tmp_path / "boundary.h5"
    write_handoff(
        boundary,
        envelope,
        bank,
        provenance={
            "parastell_commit": provenance["parastell_commit"],
            "openmc_version": provenance["openmc_version"],
            "openmc_statepoint_sha256": provenance["statepoint_sha256"],
            "dagmc_geometry_sha256": geometry["raw_h5m_sha256"],
            "source_definition_sha256": source["source_definition_sha256"],
            "histories": provenance["histories"],
        },
        normalization={
            "basis": "per source history",
            "particles_per_source_history": 1.0,
            "area_basis": "surface integrated",
            "energy_bin_width": "group integrated",
            "solid_angle_measure": "angle integrated",
            "quantity": "partial crossing current",
            "time_basis": "none",
            "physical_source_rate_per_s": source["physical_source_rate_per_s"],
        },
    )
    damage_gas = tmp_path / "damage-gas.h5"
    product_provenance = {
        **provenance,
        **geometry,
        "source_definition_sha256": source["source_definition_sha256"],
        "nuclear_data_manifest_sha256": nuclear_data["manifest_sha256"],
    }
    export_magnet_damage_gas(
        _damage_gas_statepoint(),
        damage_gas,
        tally_inventory=_damage_gas_inventory(),
        cell_magnet_ids={88: "magnet-A"},
        cell_volumes_cm3={88: 2.0},
        physical_source_rate_per_s=source["physical_source_rate_per_s"],
        provenance=product_provenance,
    )
    return {
        "provenance": provenance,
        "geometry": geometry,
        "source": source,
        "nuclear_data": nuclear_data,
        "materials": materials,
        "magnet_inventory": [{"magnet_id": "magnet-A"}],
        "products": [
            {
                "kind": "volume_scalar_flux",
                "magnet_id": "all-selected-magnets",
                "path": scalar,
                "units": "particles/cm2/s",
                "normalization": "physical_source_rate",
                "quantity": "volume_scalar_flux",
            },
            {
                "kind": "boundary_phase_space",
                "magnet_id": "magnet-A",
                "path": boundary,
                "units": "crossings/source",
                "normalization": "per_source_history",
                "quantity": "partial_crossing_current",
                "record_count": 1,
            },
            {
                "kind": "damage_and_gas_production",
                "magnet_id": "all-selected-magnets",
                "path": damage_gas,
                "units": "mixed_explicit_in_product",
                "normalization": "physical_source_rate",
                "quantity": "damage_energy_and_gas_production",
            },
        ],
        "verification": {"status": "PASS"},
    }


def _write_bundle(tmp_path, values):
    bundle = tmp_path / "bundle"
    write_radiation_field_bundle(bundle, **values)
    return bundle


def _update_product_hash(bundle, product_index):
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    product = manifest["products"][product_index]
    path = bundle / product["path"]
    product["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    product["size_bytes"] = path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_neutral_bundle_round_trip_and_hash_validation(
    tmp_path, monkeypatch, bundle_inputs
):
    bundle = _write_bundle(tmp_path, bundle_inputs)
    monkeypatch.setitem(sys.modules, "openmc", None)
    monkeypatch.setitem(sys.modules, "pydagmc", None)
    restored = read_radiation_field_bundle(bundle)
    assert restored["schema"].endswith("/v1.0.0")
    assert len(restored["products"]) == 3
    assert restored["bank_completeness"] == {
        "status": "COMPLETE",
        "bank_count": 1,
        "classification_counts": {
            "COMPLETE_CROSSING_BANK": 1,
            "SAMPLED_CROSSING_BANK": 0,
            "TRUNCATED_INVALID_BANK": 0,
        },
        "record_count_total": 1,
        "empty_bank_count": 0,
        "all_banks_valid": True,
        "all_banks_complete": True,
        "empty_banks_are_physical_zero": False,
    }
    product = bundle / restored["products"][0]["path"]
    product.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_radiation_field_bundle(bundle)


def test_bundle_requires_one_boundary_product_per_inventory_magnet(
    tmp_path, bundle_inputs
):
    bundle_inputs["magnet_inventory"].append({"magnet_id": "magnet-B"})
    with pytest.raises(ValueError, match="cover every inventory magnet"):
        write_radiation_field_bundle(tmp_path / "bundle", **bundle_inputs)


def test_bundle_reader_rejects_incomplete_boundary_coverage(
    tmp_path, bundle_inputs
):
    bundle = _write_bundle(tmp_path, bundle_inputs)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["magnet_inventory"].append({"magnet_id": "magnet-B"})
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cover every inventory magnet"):
        read_radiation_field_bundle(bundle)


def test_scalar_flux_content_is_validated_after_hash_binding(
    tmp_path, bundle_inputs
):
    bundle = _write_bundle(tmp_path, bundle_inputs)
    scalar = bundle / "volume_flux/all-selected-magnets/scalar.h5"
    with h5py.File(scalar, "r+") as handle:
        handle["scalar_flux_fields/neutron_ccfe_709/mean_physical"][
            0, 0
        ] *= 2.0
    _update_product_hash(bundle, 0)
    with pytest.raises(ValueError, match="normalization is inconsistent"):
        read_radiation_field_bundle(bundle)


def test_boundary_completeness_is_required_inside_hdf5(
    tmp_path, bundle_inputs
):
    bundle = _write_bundle(tmp_path, bundle_inputs)
    boundary = bundle / "boundary/magnet-A/boundary.h5"
    with h5py.File(boundary, "r+") as handle:
        manifest = json.loads(handle["manifest_json"].asstr()[()])
        del manifest["bank_metadata"]["surface_bank_completeness"]
        handle["manifest_json"][()] = json.dumps(manifest, sort_keys=True)
    _update_product_hash(bundle, 1)
    with pytest.raises(ValueError, match="completeness is absent"):
        read_radiation_field_bundle(bundle)


def test_top_level_bank_completeness_is_required_and_recomputed(
    tmp_path, bundle_inputs
):
    bundle = _write_bundle(tmp_path, bundle_inputs)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["bank_completeness"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="bank_completeness"):
        read_radiation_field_bundle(bundle)

    bundle = _write_bundle(tmp_path / "second", bundle_inputs)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bank_completeness"]["record_count_total"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="bank-completeness summary"):
        read_radiation_field_bundle(bundle)


def test_bundle_provenance_is_fail_closed_and_accepts_seed_alias(
    tmp_path, bundle_inputs
):
    bundle_inputs["provenance"]["seed"] = bundle_inputs["provenance"].pop(
        "seeds"
    )[0]
    bundle = _write_bundle(tmp_path, bundle_inputs)
    assert read_radiation_field_bundle(bundle)["provenance"]["seeds"] == [17]

    del bundle_inputs["provenance"]["vmec_sha256"]
    with pytest.raises(ValueError, match="vmec_sha256"):
        write_radiation_field_bundle(tmp_path / "missing", **bundle_inputs)


def test_arbitrary_hdf5_bytes_are_not_accepted(tmp_path, bundle_inputs):
    bundle_inputs["products"][0]["path"].write_bytes(b"not hdf5")
    with pytest.raises(ValueError, match="not valid HDF5"):
        write_radiation_field_bundle(tmp_path / "bad", **bundle_inputs)


def test_scalar_flux_requires_full_provenance_and_canonical_route(
    tmp_path, bundle_inputs
):
    bundle = _write_bundle(tmp_path, bundle_inputs)
    scalar = bundle / "volume_flux/all-selected-magnets/scalar.h5"
    with h5py.File(scalar, "r+") as handle:
        manifest = json.loads(handle["manifest_json"].asstr()[()])
        del manifest["provenance"]["openmc_commit"]
        handle["manifest_json"][()] = json.dumps(manifest, sort_keys=True)
    _update_product_hash(bundle, 0)
    with pytest.raises(ValueError, match="openmc_commit"):
        read_radiation_field_bundle(bundle)

    bundle = _write_bundle(tmp_path / "route", bundle_inputs)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["products"][0]["magnet_id"] = "magnet-A"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="path disagrees"):
        read_radiation_field_bundle(bundle)


def test_damage_gas_is_bound_to_bundle_provenance_and_magnet_inventory(
    tmp_path, bundle_inputs
):
    bundle = _write_bundle(tmp_path, bundle_inputs)
    damage = (
        bundle
        / "damage_and_gas_production"
        / "all-selected-magnets"
        / "damage-gas.h5"
    )
    with h5py.File(damage, "r+") as handle:
        manifest = json.loads(handle["manifest_json"].asstr()[()])
        manifest["provenance"]["statepoint_sha256"] = "0" * 64
        handle["manifest_json"][()] = json.dumps(manifest, sort_keys=True)
    _update_product_hash(bundle, 2)
    with pytest.raises(ValueError, match="statepoint_sha256"):
        read_radiation_field_bundle(bundle)

    bundle = _write_bundle(tmp_path / "magnet", bundle_inputs)
    damage = (
        bundle
        / "damage_and_gas_production"
        / "all-selected-magnets"
        / "damage-gas.h5"
    )
    with h5py.File(damage, "r+") as handle:
        handle["damage_energy/neutron/magnet_ids"][0] = "magnet-B"
        handle["gas_production/neutron/magnet_ids"][0] = "magnet-B"
    _update_product_hash(bundle, 2)
    with pytest.raises(ValueError, match="absent from the inventory"):
        read_radiation_field_bundle(bundle)


def test_damage_and_gas_product_contract_is_explicitly_not_dpa_or_appm():
    product = {
        "kind": "damage_and_gas_production",
        "magnet_id": "all-selected-magnets",
        "path": "damage-and-gas.h5",
        "units": "mixed_explicit_in_product",
        "normalization": "physical_source_rate",
        "quantity": "damage_energy_and_gas_production",
    }
    _validate_product(product)
    with pytest.raises(ValueError, match="unknown quantity"):
        _validate_product({**product, "quantity": "dpa_and_appm"})
