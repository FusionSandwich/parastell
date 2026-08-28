import copy

import numpy as np
import pytest

from parastell.boundary_phase_space_figures import summarize_phase_space
from parastell.boundary_phase_space_figures import validate_figure_inputs
from parastell.boundary_phase_space_figures import write_phase_space_figures


def _records():
    return {
        "position_global_cm": np.asarray(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
        ),
        "position_local_cm": np.asarray(
            [[0.1, 0.2, 0.0], [0.2, 0.3, 0.0], [0.3, 0.4, 0.0]]
        ),
        "direction_global": np.asarray(
            [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        ),
        "outward_normal_global": np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        "energy_eV": np.asarray([14.1e6, 1.0e6, 1.0]),
        "time_s": np.asarray([0.0, 1.0e-9, 2.0e-9]),
        "weight": np.asarray([1.0, 2.0, 3.0]),
        "particle": np.asarray([2112, 2112, 22]),
        "surface_id": np.asarray([11, 11, 12]),
        "mu": np.asarray([-1.0, 1.0, 0.0]),
        "facet_id": np.asarray([7, 8, 9]),
        "barycentric_coordinates": np.asarray(
            [[0.2, 0.3, 0.5], [0.1, 0.1, 0.8], [0.0, 0.5, 0.5]]
        ),
    }


def test_phase_space_validation_preserves_all_required_correlations():
    values = validate_figure_inputs(_records())
    summary = summarize_phase_space(
        values, source_histories=10, grazing_tolerance=1.0e-8
    )

    assert summary["record_count"] == 3
    assert summary["incoming_count"] == 1
    assert summary["outgoing_count"] == 1
    assert summary["grazing_count"] == 1
    assert summary["per_source_weight_sum"] == pytest.approx(0.6)
    assert summary["facet_localization_present"] is True
    assert summary["barycentric_localization_present"] is True
    assert summary["local_coordinates_present"] is True


def test_phase_space_validation_accepts_root_auditor_column_names():
    records = _records()
    records["openmc_weight"] = records.pop("weight")
    records["particle_pdg"] = records.pop("particle")

    values = validate_figure_inputs(records)

    assert values.weight.tolist() == [1.0, 2.0, 3.0]
    assert values.particle.tolist() == [2112, 2112, 22]


def test_phase_space_validation_rejects_mu_inconsistent_with_normal_sense():
    records = copy.deepcopy(_records())
    records["mu"][0] = 1.0

    with pytest.raises(ValueError, match="stored mu"):
        validate_figure_inputs(records)


def test_phase_space_validation_rejects_invalid_barycentric_coordinates():
    records = copy.deepcopy(_records())
    records["barycentric_coordinates"][0] = [-0.1, 0.4, 0.7]

    with pytest.raises(ValueError, match="barycentric"):
        validate_figure_inputs(records)


def test_matched_figures_are_hash_bound(tmp_path):
    manifest = write_phase_space_figures(
        _records(),
        tmp_path,
        geometry_label="synthetic-variant",
        geometry_sha256="a" * 64,
        source_bank_sha256="b" * 64,
        source_histories=10,
    )

    assert len(manifest["figures"]) == 4
    assert all(
        (tmp_path / row["path"]).is_file() for row in manifest["figures"]
    )
    assert (tmp_path / "FIGURE_MANIFEST.json").is_file()
