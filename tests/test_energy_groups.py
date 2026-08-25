import json

import pytest

from parastell.energy_groups import compare_structures
from parastell.energy_groups import get_structure
from parastell.energy_groups import list_structures
from parastell.energy_groups import load_custom_structure
from parastell.energy_groups import validate_edges
from parastell.energy_groups.cli import main


@pytest.mark.parametrize(
    ("name", "particle", "groups", "minimum", "maximum"),
    [
        ("smoke-7", "neutron", 7, 0.0, 2.0e7),
        ("regression-182", "neutron", 182, 0.0, 2.0e7),
        ("CCFE-709", "neutron", 709, 1.0e-5, 1.0e9),
        ("UKAEA-1102", "neutron", 1102, 1.0e-5, 1.0e9),
        ("smoke-42", "photon", 42, 1.0e3, 3.0e7),
        ("CCFE-162", "photon", 162, 5.0e3, 1.0e9),
    ],
)
def test_exact_built_in_structure(name, particle, groups, minimum, maximum):
    structure = get_structure(name, particle=particle)
    assert structure.group_count == groups
    assert structure.edges_eV[0] == minimum
    assert structure.edges_eV[-1] == maximum
    assert structure.edge_sha256
    assert structure.local_file_sha256


def test_continuous_photon_is_not_a_fictitious_group_grid():
    structure = get_structure("continuous-photon")
    assert structure.is_continuous
    assert structure.group_count is None
    assert structure.edges_eV is None


def test_registry_names_and_aliases_are_stable():
    names = {structure.name for structure in list_structures()}
    assert {"CCFE-709", "UKAEA-1102", "CCFE-162"} <= names
    assert get_structure("PSTL-SOFTWARE-175+").name == "regression-182"


def test_particle_semantics_are_enforced():
    with pytest.raises(ValueError, match="not photon"):
        get_structure("CCFE-709", particle="photon")


@pytest.mark.parametrize(
    "edges", ([1.0], [0.0, 1.0, 1.0], [0.0, float("inf")])
)
def test_invalid_edges_are_rejected(edges):
    with pytest.raises(ValueError):
        validate_edges(edges)


def test_custom_descending_csv_round_trip(tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text("20, 10, 1\n", encoding="ascii")
    structure = load_custom_structure(path, particle="photon", units="MeV")
    assert structure.edges_eV == (1.0e6, 1.0e7, 2.0e7)
    assert structure.group_count == 2


def test_compare_does_not_claim_response_equivalence():
    result = compare_structures("CCFE-709", "UKAEA-1102")
    assert result["particles_match"] is True
    assert result["response_equivalence_assessed"] is False
    assert result["shared_exact_boundaries"] > 2


def test_cli_list_inspect_validate_compare(tmp_path, capsys):
    assert main(["list"]) == 0
    assert "CCFE-709" in capsys.readouterr().out
    assert main(["inspect", "CCFE-162", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["particle"] == "photon"
    custom = tmp_path / "groups.csv"
    custom.write_text("0, 1, 2", encoding="ascii")
    assert main(["validate", str(custom), "--particle", "neutron"]) == 0
    assert json.loads(capsys.readouterr().out)["group_count"] == 2
    assert main(["compare", "CCFE-709", "UKAEA-1102"]) == 0
    assert json.loads(capsys.readouterr().out)["first_groups"] == 709
