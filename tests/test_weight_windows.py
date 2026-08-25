import json

import pytest

from parastell.weight_windows import (
    WeightWindowArtifactContract,
    candidate_neutron_weight_window_grids,
    qualify_weight_windows,
    require_compatible_weight_window,
    validate_split_controls,
    validate_weight_window_energy_grid,
    weight_window_disabled_fallback,
    write_weight_window_contract,
)


def response(name, std, mean=1.0, particle="neutron"):
    return {
        "response_id": name,
        "definition_hash": "a" * 64,
        "run_contract_sha256": "b" * 64,
        "mean": mean,
        "std_dev": std,
        "wall_time_s": 10.0,
        "effective_sample_size": 100.0 / std,
        "seed_count": 3,
        "particle": particle,
        "primary": True,
        "critical": True,
    }


def contract(seed=11):
    return WeightWindowArtifactContract(
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
        generator_settings={"update_interval": 1},
        openmc_version="0.16.0",
        openmc_source_sha="8" * 40,
        generation_histories=1000,
        generation_batches=10,
        generation_seed=seed,
        selected_magnet_ids=("magnet-A",),
    )


def test_ww_energy_grid_split_controls_and_disabled_fallback():
    candidates = candidate_neutron_weight_window_grids()
    assert {len(edges) - 1 for edges in candidates.values()} == {4, 8, 16}
    with pytest.raises(ValueError, match="transport range"):
        validate_weight_window_energy_grid(
            [1.0, 20.0e6], transport_min_eV=0.0, transport_max_eV=20.0e6
        )
    assert (
        validate_split_controls(max_history_splits=1000, max_split=10)[
            "historical_unqualified_100000_setting_used"
        ]
        is False
    )
    with pytest.raises(ValueError, match="20000"):
        validate_split_controls(max_history_splits=100000, max_split=10)
    assert (
        weight_window_disabled_fallback("no material benefit")[
            "production_transport"
        ]
        == "UNBIASED"
    )


def test_ww_qualification_accepts_fom_gain_and_rejects_bias():
    unbiased = [response("flux", 0.2), response("heating", 0.3)]
    improved = [response("flux", 0.05), response("heating", 0.1)]
    result = qualify_weight_windows(unbiased, improved)
    assert result["classification"] == "QUALIFIED_AND_ENABLED"
    assert result["weight_windows_enabled"]
    biased = [response("flux", 0.01, 2.0), response("heating", 0.01, 2.0)]
    result = qualify_weight_windows(unbiased, biased)
    assert result["classification"] == "REJECTED_BIAS"
    assert result["multiple_comparison_method"] == "Benjamini-Hochberg"


def test_weight_window_artifact_contract_refuses_stale_inputs(tmp_path):
    artifact = tmp_path / "weight_windows.h5"
    artifact.write_bytes(b"weight-window")
    path = tmp_path / "contract.json"
    write_weight_window_contract(path, contract(), weight_window_path=artifact)
    assert require_compatible_weight_window(path, contract())[
        "weight_window_artifact"
    ]["size_bytes"] == len(b"weight-window")
    with pytest.raises(ValueError, match="contract mismatch"):
        require_compatible_weight_window(path, contract(seed=12))
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["contract_sha256"] == contract().sha256
