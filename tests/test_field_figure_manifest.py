import pytest

from parastell.qualification_figures import FIGURE_FILENAMES
from parastell.qualification_figures import validate_figure_manifest


def test_field_figure_requires_uncertainty_semantics():
    rows = []
    for filename in FIGURE_FILENAMES:
        rows.append(
            {
                "filename": filename,
                "sha256": "a" * 64,
                "units": "unit",
                "normalization_basis": "basis",
                "geometry_sha256": "b" * 64,
                "source_mesh_sha256": "c" * 64,
                "run_hash": "d" * 64,
                "particle": "neutron",
                "energy_integration_range_eV": "0-20MeV",
                "uncertainty_meaning": "one sigma",
                "simulation_status": "direct",
                "data_status": "available",
            }
        )
    rows[12]["uncertainty_meaning"] = ""
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_figure_manifest({"figures": rows})
