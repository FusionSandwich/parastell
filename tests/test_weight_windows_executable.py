import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from parastell.weight_windows import (
    WeightWindowArtifactContract,
    aggregate_weight_window_campaign_rows,
    qualify_weight_windows,
    require_compatible_weight_window,
    validate_weight_window_hdf5,
    write_weight_window_contract,
)


def _contract(**changes):
    value = WeightWindowArtifactContract(
        canonical_geometry_fingerprint="1" * 64,
        raw_h5m_sha256="2" * 64,
        source_definition_sha256="3" * 64,
        source_mesh_sha256="4" * 64,
        physical_source_rate_per_s=1.0e20,
        material_manifest_sha256="5" * 64,
        nuclear_data_manifest_sha256="6" * 64,
        weight_window_mesh_sha256="7" * 64,
        particle_type="neutron",
        energy_bounds_eV=(0.0, 1.0e5, 13.0e6, 15.0e6, 20.0e6),
        generator_method="magic",
        generator_settings={
            "update_interval": 1,
            "max_split": 10,
            "survival_ratio": 3.0,
            "max_lower_bound_ratio": None,
            "weight_cutoff": 1.0e-38,
        },
        openmc_version="0.16.0",
        openmc_source_sha="8" * 40,
        generation_histories=1000,
        generation_batches=10,
        generation_seed=11,
        selected_magnet_ids=("magnet-A",),
    )
    return replace(value, **changes)


def _window(**changes):
    value = {
        "particle_type": SimpleNamespace(name="NEUTRON"),
        "energy_bounds": np.asarray((0.0, 1.0e5, 13.0e6, 15.0e6, 20.0e6)),
        "lower_ww_bounds": np.asarray(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        ),
        "upper_ww_bounds": np.asarray(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        ),
        "mesh": SimpleNamespace(n_elements=2),
        "max_split": 10,
        "survival_ratio": 3.0,
        "max_lower_bound_ratio": None,
        "weight_cutoff": 1.0e-38,
    }
    value.update(changes)
    return SimpleNamespace(**value)


def _mock_openmc(monkeypatch, windows):
    class WeightWindowsList:
        @staticmethod
        def from_hdf5(path):
            assert path.is_file()
            return windows

    monkeypatch.setitem(
        sys.modules,
        "openmc",
        SimpleNamespace(WeightWindowsList=WeightWindowsList),
    )


def test_semantic_hdf_validation_checks_bounds_coverage_and_contract(
    monkeypatch, tmp_path
):
    artifact = tmp_path / "weight_windows.h5"
    artifact.write_bytes(b"mock HDF5")
    _mock_openmc(monkeypatch, [_window()])
    result = validate_weight_window_hdf5(
        artifact, expected_contract=_contract()
    )
    assert result == {
        "status": "PASS",
        "particle_type": "neutron",
        "energy_bounds_eV": [0.0, 1.0e5, 13.0e6, 15.0e6, 20.0e6],
        "energy_bin_count": 4,
        "mesh_element_count": 2,
        "weight_window_bin_count": 8,
        "nonzero_lower_bound_count": 8,
        "finite_nonnegative_bounds": True,
        "max_split": 10,
        "survival_ratio": 3.0,
        "max_lower_bound_ratio": None,
        "weight_cutoff": 1.0e-38,
    }


@pytest.mark.parametrize(
    ("window", "message"),
    [
        (_window(lower_ww_bounds=np.zeros((2, 4))), "zero populated"),
        (_window(mesh=SimpleNamespace(n_elements=3)), "cover every mesh"),
        (
            _window(particle_type=SimpleNamespace(name="PHOTON")),
            "particle type is stale",
        ),
        (_window(max_split=20), "max_split control is stale"),
    ],
)
def test_semantic_hdf_validation_rejects_stale_or_empty_artifacts(
    monkeypatch, tmp_path, window, message
):
    artifact = tmp_path / "weight_windows.h5"
    artifact.write_bytes(b"mock HDF5")
    _mock_openmc(monkeypatch, [window])
    with pytest.raises(ValueError, match=message):
        validate_weight_window_hdf5(artifact, expected_contract=_contract())


def test_contract_rejects_tampering_escape_and_stale_production_input(
    tmp_path,
):
    contract_directory = tmp_path / "contracts"
    contract_directory.mkdir()
    artifact = contract_directory / "weight_windows.h5"
    artifact.write_bytes(b"first artifact")
    contract_path = contract_directory / "contract.json"
    write_weight_window_contract(
        contract_path, _contract(), weight_window_path=artifact
    )
    compatible = require_compatible_weight_window(contract_path, _contract())
    assert compatible["resolved_weight_window_artifact_path"] == str(
        artifact.resolve()
    )

    artifact.write_bytes(b"tampered artifact")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        require_compatible_weight_window(contract_path, _contract())
    artifact.write_bytes(b"first artifact")

    with pytest.raises(ValueError, match="contract mismatch"):
        require_compatible_weight_window(
            contract_path,
            replace(_contract(), canonical_geometry_fingerprint="9" * 64),
        )

    value = json.loads(contract_path.read_text(encoding="utf-8"))
    value["generation_seed"] = 99
    contract_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="stored weight-window contract hash"):
        require_compatible_weight_window(contract_path, _contract())
    write_weight_window_contract(
        contract_path, _contract(), weight_window_path=artifact
    )

    outside = tmp_path / "outside"
    outside.mkdir()
    escaped_artifact = outside / "weight_windows.h5"
    escaped_artifact.write_bytes(b"first artifact")
    value = json.loads(contract_path.read_text(encoding="utf-8"))
    value["weight_window_artifact"]["path"] = "../outside/weight_windows.h5"
    contract_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes the contract directory"):
        require_compatible_weight_window(contract_path, _contract())


def _campaign_row(variant, response_id, seed, *, estimate=1.0):
    return {
        "variant": variant,
        "seed": seed,
        "response_id": response_id,
        "definition_hash": ("a" if response_id == "flux" else "b") * 64,
        "run_contract_sha256": "c" * 64,
        "particle": "neutron",
        "primary": True,
        "critical": response_id == "flux",
        "estimate": estimate,
        "within_run_std_dev": 0.1,
        "wall_time_s": 2.0,
        "effective_sample_size": 10.0,
    }


def _campaign_rows():
    return [
        _campaign_row(variant, response_id, seed, estimate=1.0 + seed / 100.0)
        for variant in ("unbiased", "weight_window")
        for seed in (101, 102, 103)
        for response_id in ("flux", "heating")
    ]


def test_three_distinct_paired_seeds_are_aggregated_conservatively():
    result = aggregate_weight_window_campaign_rows(_campaign_rows())
    for variant in ("unbiased", "weight_window"):
        for response in result[variant]:
            assert response["seed_count"] == 3
            assert response["seeds"] == [101, 102, 103]
            assert response["std_dev"] == max(
                response["observed_between_seed_standard_error"],
                response["propagated_within_run_standard_error"],
            )
            assert response["effective_sample_size"] == 30.0
    qualification = qualify_weight_windows(
        result["unbiased"], result["weight_window"]
    )
    assert qualification["classification"] == "NO_MATERIAL_BENEFIT_DISABLE"
    assert qualification["run_contract_sha256"] == "c" * 64


def test_aggregation_rejects_duplicate_and_unpaired_seed_inventories():
    duplicate = _campaign_rows()
    for row in duplicate:
        if row["variant"] == "weight_window" and row["seed"] == 103:
            row["seed"] = 102
    with pytest.raises(ValueError, match="distinct positive seeds"):
        aggregate_weight_window_campaign_rows(duplicate)

    unpaired = _campaign_rows()
    for row in unpaired:
        if row["variant"] == "weight_window":
            row["seed"] += 10
    with pytest.raises(ValueError, match="paired seed inventories"):
        aggregate_weight_window_campaign_rows(unpaired)


def _response(response_id, particle, std_dev):
    return {
        "response_id": response_id,
        "definition_hash": ("d" if particle == "neutron" else "e") * 64,
        "run_contract_sha256": "f" * 64,
        "particle": particle,
        "primary": True,
        "critical": True,
        "mean": 1.0,
        "std_dev": std_dev,
        "wall_time_s": 10.0,
        "seed_count": 3,
    }


def test_photon_only_improvement_cannot_qualify_neutron_artifact():
    unbiased = [
        _response("neutron-flux", "neutron", 0.2),
        _response("photon-flux", "photon", 0.2),
    ]
    candidate = [
        _response("neutron-flux", "neutron", 0.2),
        _response("photon-flux", "photon", 0.05),
    ]
    neutron = qualify_weight_windows(
        unbiased, candidate, artifact_particle_type="neutron"
    )
    assert neutron["particle_decisions"]["photon"]["classification"] == (
        "QUALIFIED_AND_ENABLED"
    )
    assert neutron["particle_decisions"]["neutron"]["classification"] == (
        "NO_MATERIAL_BENEFIT_DISABLE"
    )
    assert neutron["classification"] == "NO_MATERIAL_BENEFIT_DISABLE"
    assert not neutron["weight_windows_enabled"]

    photon = qualify_weight_windows(
        unbiased, candidate, artifact_particle_type="photon"
    )
    assert photon["classification"] == "QUALIFIED_AND_ENABLED"
    assert photon["weight_windows_enabled"]
