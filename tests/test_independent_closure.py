import json

import h5py
import pytest

from parastell.independent_closure import compare_independent_handoffs


def _write_handoff(path, *, bank, tally, bank_sigma=0.1, tally_sigma=0.1):
    entry = {
        "bank_current_per_source": bank,
        "tally_current_per_source": tally,
        "bank_poisson_std_dev": bank_sigma,
        "tally_std_dev": tally_sigma,
    }
    senses = {
        name: dict(entry)
        for name in ("incoming", "outgoing", "net", "total_crossing")
    }
    manifest = {
        "schema": "parastell.magnet_boundary_source/v2.1.0",
        "envelope": {
            "dagmc_geometry_sha256": "geometry",
            "envelope_id": "envelope",
            "magnet_component": "magnet",
        },
        "provenance": {"source_definition_sha256": "source"},
        "bank_metadata": {
            "same_run_integrity_closure": {
                "neutron": {
                    "whole_envelope": senses,
                    "by_surface": [{"surface_id": 1, **senses}],
                }
            }
        },
    }
    with h5py.File(path, "w") as handle:
        handle.create_dataset("manifest_json", data=json.dumps(manifest))


def test_independent_closure_accepts_statistically_consistent_runs(tmp_path):
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _write_handoff(first, bank=1.0, tally=1.1)
    _write_handoff(second, bank=0.9, tally=1.0)
    report = compare_independent_handoffs(
        first, second, reference_seed=1, replicate_seed=2
    )
    assert report["passes"]
    assert report["comparison_count"] == 16


def test_independent_closure_rejects_same_seed_and_inconsistent_runs(tmp_path):
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _write_handoff(
        first, bank=1.0, tally=1.0, bank_sigma=0.01, tally_sigma=0.01
    )
    _write_handoff(
        second, bank=2.0, tally=2.0, bank_sigma=0.01, tally_sigma=0.01
    )
    with pytest.raises(ValueError, match="distinct random seeds"):
        compare_independent_handoffs(
            first, second, reference_seed=3, replicate_seed=3
        )
    report = compare_independent_handoffs(
        first, second, reference_seed=3, replicate_seed=4
    )
    assert not report["passes"]
    assert report["failure_count"] == report["comparison_count"]
