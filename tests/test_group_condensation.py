import json

import numpy as np
import pytest

from parastell.energy_groups import load_custom_structure
from parastell.group_condensation import condense_groups


def problem():
    edges = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    spectra = {
        "soft": [4.0, 3.0, 2.0, 1.0, 0.5, 0.25],
        "hard": [0.25, 0.5, 1.0, 2.0, 3.0, 4.0],
    }
    responses = {
        "heating": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
        "capture": [5.0, 4.0, 3.0, 0.3, 0.2, 0.1],
    }
    return edges, spectra, responses


def test_condensation_conserves_every_spectrum_and_protected_boundary():
    edges, spectra, responses = problem()
    result = condense_groups(
        edges,
        spectra,
        responses,
        protected_boundaries_eV=[3.0],
        relative_tolerance=0.02,
        min_groups=2,
        max_groups=5,
    )
    assert 3.0 in result.edges_eV
    for name, values in spectra.items():
        assert sum(result.collapsed_spectra[name]) == pytest.approx(
            sum(values)
        )
    assert result.maximum_relative_error <= 0.02
    assert result.qualified


def test_merge_history_is_deterministic():
    args = problem()
    first = condense_groups(
        *args, relative_tolerance=0.05, min_groups=2, max_groups=4
    )
    second = condense_groups(
        *args, relative_tolerance=0.05, min_groups=2, max_groups=4
    )
    assert first.edges_eV == second.edges_eV
    assert first.merge_history == second.merge_history


def test_all_spectra_and_responses_are_protected():
    result = condense_groups(
        *problem(), relative_tolerance=0.01, min_groups=2, max_groups=6
    )
    assert set(result.relative_errors) == {"hard", "soft"}
    assert set(result.relative_errors["hard"]) == {"capture", "heating"}
    assert all(
        error <= 0.01
        for values in result.relative_errors.values()
        for error in values.values()
    )


def test_absent_protected_boundary_is_rejected():
    with pytest.raises(ValueError, match="absent"):
        condense_groups(
            *problem(),
            protected_boundaries_eV=[2.45],
            relative_tolerance=0.1,
            max_groups=6,
        )


def test_unqualified_result_cannot_be_registered(tmp_path):
    result = condense_groups(
        *problem(), relative_tolerance=0.0, min_groups=1, max_groups=1
    )
    assert not result.qualified
    with pytest.raises(RuntimeError, match="unqualified"):
        result.write_registry_candidate(
            tmp_path / "bad.json",
            name="response-selected-neutron-v1",
            particle="neutron",
            provenance={},
        )


def test_qualified_registry_candidate_round_trip(tmp_path):
    result = condense_groups(
        *problem(), relative_tolerance=0.05, min_groups=2, max_groups=5
    )
    path = tmp_path / "selected.json"
    result.write_registry_candidate(
        path,
        name="response-selected-neutron-test",
        particle="neutron",
        provenance={"study": "synthetic-test"},
    )
    structure = load_custom_structure(path, particle="neutron")
    assert structure.edges_eV == result.edges_eV
    assert json.loads(path.read_text())["status"] == "response-qualified"
