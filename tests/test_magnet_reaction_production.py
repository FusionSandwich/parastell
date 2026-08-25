from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import parastell.magnet_radiation_field_bundle as bundle_module
from parastell.magnet_reaction_production import (
    export_magnet_reaction_production,
)
from parastell.magnet_reaction_production import (
    validate_magnet_reaction_production,
)
from parastell.magnet_radiation_field_bundle import (
    write_radiation_field_bundle,
)


def _filter(name, bins):
    cls = type(name, (), {})
    value = cls()
    value.bins = bins
    value.num_bins = len(bins)
    return value


def _tally(name, response_filter, response_bins, response_count):
    filters = [
        _filter("CellFilter", [20, 22]),
        _filter("ParticleFilter", ["neutron"]),
        _filter("EnergyFilter", [(0.0, 1.0), (1.0, 2.0)]),
        _filter(response_filter, response_bins),
    ]
    size = 2 * 1 * 2 * response_count
    return SimpleNamespace(
        name=name,
        filters=filters,
        num_nuclides=1,
        num_scores=1,
        mean=np.arange(size, dtype=float).reshape((-1, 1, 1)) / 100.0,
        std_dev=np.full((size, 1, 1), 0.001),
    )


def _statepoint():
    tallies = {
        "pstl_magnet_neutron_reactions": _tally(
            "pstl_magnet_neutron_reactions",
            "ReactionFilter",
            ["(n,elastic)", "(n,gamma)"],
            2,
        )
    }
    for particle in ("photon", "neutron", "electron", "positron"):
        bins = [(particle, 0.0, 1.0), (particle, 1.0, 2.0)]
        name = f"pstl_magnet_production_{particle}"
        tallies[name] = _tally(name, "ParticleProductionFilter", bins, 2)
    return SimpleNamespace(get_tally=lambda *, name: tallies[name])


def test_reaction_production_round_trip_and_physical_scaling(tmp_path):
    path = export_magnet_reaction_production(
        _statepoint(),
        tmp_path / "product.h5",
        magnet_ids=["magnet-20", "magnet-22"],
        physical_source_rate_per_s=10.0,
    )
    summary = validate_magnet_reaction_production(path)
    assert summary["magnets"] == 2
    assert set(summary["production_events_per_source"]) == {
        "photon",
        "neutron",
        "electron",
        "positron",
    }
    with h5py.File(path, "r") as stream:
        photon = stream["production/photon"]
        assert photon.attrs["transported_by_openmc"]
        assert photon["mean_events_per_s"][...] == pytest.approx(
            photon["mean_events_per_source"][...] * 10.0
        )
        assert not stream["production/electron"].attrs["transported_by_openmc"]


def test_noncontiguous_outgoing_energy_bins_are_rejected(tmp_path):
    statepoint = _statepoint()
    tally = statepoint.get_tally(name="pstl_magnet_production_photon")
    tally.filters[-1].bins = [("photon", 0.0, 1.0), ("photon", 1.5, 2.0)]
    with pytest.raises(ValueError, match="contiguous"):
        export_magnet_reaction_production(statepoint, tmp_path / "bad.h5")


def test_invalid_physical_source_rate_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="source rate"):
        export_magnet_reaction_production(
            _statepoint(), tmp_path / "bad.h5", physical_source_rate_per_s=0.0
        )


def test_reaction_production_is_discoverable_in_bundle(tmp_path, monkeypatch):
    product = export_magnet_reaction_production(
        _statepoint(), tmp_path / "reaction.h5"
    )
    scalar = tmp_path / "scalar.h5"
    boundary = tmp_path / "boundary.h5"
    scalar.write_bytes(b"scalar-fixture")
    boundary.write_bytes(b"boundary-fixture")
    monkeypatch.setattr(
        bundle_module,
        "_validate_scalar_flux_hdf5",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bundle_module,
        "_validate_boundary_hdf5",
        lambda *args, **kwargs: {
            "classification": "COMPLETE_CROSSING_BANK",
            "record_count": 1,
            "empty": False,
        },
    )
    bundle = tmp_path / "bundle"
    manifest = write_radiation_field_bundle(
        bundle,
        provenance={
            "parastell_commit": "a" * 40,
            "openmc_version": "0.16.0",
            "openmc_commit": "b" * 40,
            "statepoint_sha256": "c" * 64,
            "vmec_sha256": "d" * 64,
            "coils_sha256": "e" * 64,
            "source_mesh_sha256": "f" * 64,
            "histories": 100,
            "batches": 10,
            "seeds": [7],
        },
        geometry={
            "raw_h5m_sha256": "1" * 64,
            "canonical_geometry_fingerprint": "2" * 64,
        },
        source={
            "source_definition_sha256": "3" * 64,
            "source_mesh_sha256": "f" * 64,
            "physical_source_rate_per_s": 1.0,
        },
        nuclear_data={"manifest_sha256": "4" * 64},
        materials={"resolved_manifest_sha256": "5" * 64},
        magnet_inventory=[{"magnet_id": "magnet-20", "dagmc_volume_id": 20}],
        products=[
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
                "magnet_id": "magnet-20",
                "path": boundary,
                "units": "crossings/source",
                "normalization": "per_source_history",
                "quantity": "partial_crossing_current",
            },
            {
                "kind": "reaction_production",
                "magnet_id": "all-selected-magnets",
                "path": product,
                "units": "events/source",
                "normalization": "per_source",
                "quantity": "reaction_and_particle_production",
            },
        ],
        verification={},
    )
    assert "reaction_production" in {
        item["kind"] for item in manifest["products"]
    }
