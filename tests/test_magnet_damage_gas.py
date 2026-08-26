from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from parastell.magnet_damage_gas import (
    DAMAGE_TALLY,
    GAS_OPENMC_SCORES,
    GAS_PRODUCTION_SCORES,
    GAS_TALLY,
    SCHEMA,
    export_magnet_damage_gas,
    validate_magnet_damage_gas,
)


def _filter(name, bins):
    cls = type(name, (), {})
    value = cls()
    value.bins = bins
    value.num_bins = len(bins)
    return value


def _tally(
    name, scores, *, cells=(20, 22), energy_bins=((0.0, 1.0), (1.0, 2.0))
):
    filters = [
        _filter("CellFilter", list(cells)),
        _filter("ParticleFilter", ["neutron"]),
        _filter("EnergyFilter", list(energy_bins)),
    ]
    size = len(cells) * len(energy_bins) * len(scores)
    mean = np.arange(size, dtype=float).reshape((-1, 1, len(scores))) / 10.0
    return SimpleNamespace(
        name=name,
        scores=list(scores),
        filters=filters,
        num_nuclides=1,
        num_scores=len(scores),
        mean=mean,
        std_dev=np.full_like(mean, 0.01),
    )


def _inventory(*, unavailable=()):
    unavailable = set(unavailable)
    responses = ("damage-energy", *GAS_PRODUCTION_SCORES)
    availability = {
        response: {
            "available": response not in unavailable,
            "status": (
                "AVAILABLE"
                if response not in unavailable
                else "UNAVAILABLE_IN_CONFIGURED_NUCLEAR_DATA"
            ),
        }
        for response in responses
    }
    return {
        "damage_energy": (
            None if "damage-energy" in unavailable else DAMAGE_TALLY
        ),
        "gas_production": (
            None if set(GAS_PRODUCTION_SCORES) <= unavailable else GAS_TALLY
        ),
        "response_availability": availability,
    }


def _statepoint(*, unavailable=(), gas_cells=(20, 22)):
    unavailable = set(unavailable)
    tallies = {}
    if "damage-energy" not in unavailable:
        tallies[DAMAGE_TALLY] = _tally(DAMAGE_TALLY, ["damage-energy"])
    scores = [
        score for score in GAS_PRODUCTION_SCORES if score not in unavailable
    ]
    if scores:
        tallies[GAS_TALLY] = _tally(
            GAS_TALLY,
            [GAS_OPENMC_SCORES[score] for score in scores],
            cells=gas_cells,
        )
    return SimpleNamespace(get_tally=lambda *, name: tallies[name])


def _export(tmp_path, *, unavailable=(), statepoint=None):
    return export_magnet_damage_gas(
        statepoint or _statepoint(unavailable=unavailable),
        tmp_path / "damage-gas.h5",
        tally_inventory=_inventory(unavailable=unavailable),
        cell_magnet_ids={20: "magnet-20", 22: "magnet-22"},
        cell_volumes_cm3={20: 10.0, 22: 20.0},
        physical_source_rate_per_s=5.0,
        provenance={"parastell_commit": "a" * 40},
    )


def test_damage_gas_round_trip_preserves_units_scaling_and_semantics(tmp_path):
    path = _export(tmp_path)
    summary = validate_magnet_damage_gas(path)
    assert summary["schema"] == SCHEMA
    assert summary["magnets"] == 2
    assert set(summary["gas_atoms_per_source"]) == set(GAS_PRODUCTION_SCORES)
    assert summary["unavailable_responses"] == []
    with h5py.File(path, "r") as stream:
        damage = stream["damage_energy/neutron"]
        gas = stream["gas_production/neutron"]
        assert not damage.attrs["is_dpa"]
        assert not gas.attrs["is_appm"]
        assert damage["mean_eV_per_source"].attrs["units"] == "eV/source"
        assert gas["mean_atoms_per_source"].attrs["units"] == "atoms/source"
        assert tuple(gas["openmc_score_labels"].asstr()[...]) == tuple(
            GAS_OPENMC_SCORES[score] for score in GAS_PRODUCTION_SCORES
        )
        np.testing.assert_allclose(
            gas["mean_atoms_per_s"][...],
            gas["mean_atoms_per_source"][...] * 5.0,
        )
        np.testing.assert_allclose(
            gas["mean_atoms_per_source_cm3"][0],
            gas["mean_atoms_per_source"][0] / 10.0,
        )
        assert np.isnan(damage["relative_error"][0, 0])
        assert damage["sample_status_code"][0, 0] == 0


def test_two_component_cells_can_share_one_magnet_axis_identity(tmp_path):
    path = export_magnet_damage_gas(
        _statepoint(),
        tmp_path / "paired-components.h5",
        tally_inventory=_inventory(),
        cell_magnet_ids={20: "magnet-20", 22: "magnet-20"},
        cell_volumes_cm3={20: 10.0, 22: 20.0},
        physical_source_rate_per_s=5.0,
        provenance={"parastell_commit": "a" * 40},
    )

    summary = validate_magnet_damage_gas(path)

    assert summary["magnets"] == 1
    with h5py.File(path, "r") as stream:
        assert tuple(
            stream["damage_energy/neutron/magnet_ids"].asstr()[...]
        ) == ("magnet-20", "magnet-20")


def test_unavailable_gas_response_is_absent_not_zero(tmp_path):
    unavailable = ("H2-production", "H3-production")
    path = _export(tmp_path, unavailable=unavailable)
    summary = validate_magnet_damage_gas(path)
    assert summary["unavailable_responses"] == list(unavailable)
    with h5py.File(path, "r") as stream:
        scores = tuple(
            stream["gas_production/neutron/score_labels"].asstr()[...]
        )
        assert scores == ("H1-production", "He3-production", "He4-production")
        responses = tuple(stream["response_status/response"].asstr()[...])
        statuses = tuple(stream["response_status/product_status"].asstr()[...])
        assert statuses[responses.index("H2-production")] == (
            "UNAVAILABLE_IN_CONFIGURED_NUCLEAR_DATA"
        )


def test_gas_only_product_preserves_explicit_damage_unavailability(tmp_path):
    path = _export(tmp_path, unavailable=("damage-energy",))
    summary = validate_magnet_damage_gas(path)
    assert summary["damage_energy_eV_per_source"] is None
    assert summary["unavailable_responses"] == ["damage-energy"]
    with h5py.File(path, "r") as stream:
        assert "damage_energy" not in stream
        assert "gas_production/neutron" in stream


def test_missing_or_inconsistent_inventory_fails_closed(tmp_path):
    inventory = _inventory()
    del inventory["response_availability"]["He4-production"]
    with pytest.raises(ValueError, match="missing availability status"):
        export_magnet_damage_gas(
            _statepoint(),
            tmp_path / "bad.h5",
            tally_inventory=inventory,
            cell_magnet_ids={20: "magnet-20", 22: "magnet-22"},
            cell_volumes_cm3={20: 10.0, 22: 20.0},
            physical_source_rate_per_s=5.0,
        )

    inventory = _inventory(unavailable=("damage-energy",))
    inventory["damage_energy"] = DAMAGE_TALLY
    with pytest.raises(ValueError, match="unavailable damage-energy"):
        export_magnet_damage_gas(
            _statepoint(unavailable=("damage-energy",)),
            tmp_path / "bad.h5",
            tally_inventory=inventory,
            cell_magnet_ids={20: "magnet-20", 22: "magnet-22"},
            cell_volumes_cm3={20: 10.0, 22: 20.0},
            physical_source_rate_per_s=5.0,
        )


def test_missing_tally_and_misaligned_axes_fail_closed(tmp_path):
    missing = SimpleNamespace(
        get_tally=lambda *, name: (_ for _ in ()).throw(KeyError(name))
    )
    with pytest.raises(ValueError, match="required tally"):
        _export(tmp_path, statepoint=missing)
    with pytest.raises(ValueError, match="cell axes do not match"):
        _export(tmp_path, statepoint=_statepoint(gas_cells=(20, 23)))


def test_validation_rejects_status_payload_disagreement(tmp_path):
    path = _export(tmp_path)
    with h5py.File(path, "r+") as stream:
        stream["response_status/available"][0] = False
    with pytest.raises(ValueError, match="disagrees"):
        validate_magnet_damage_gas(path)


def test_validation_rejects_inconsistent_physical_scaling(tmp_path):
    path = _export(tmp_path)
    with h5py.File(path, "r+") as stream:
        stream["gas_production/neutron/mean_atoms_per_s"][0, 0, 0] = 99.0
    with pytest.raises(ValueError, match="inconsistent normalization"):
        validate_magnet_damage_gas(path)
