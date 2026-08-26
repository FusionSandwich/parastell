import pytest

from parastell.qualification_figures import FIGURE_FILENAMES
from parastell.qualification_figures import validate_figure_manifest


def _row(filename):
    return {
        "filename": filename,
        "sha256": "a" * 64,
        "units": "cm",
        "normalization_basis": "native",
        "geometry_sha256": "b" * 64,
        "source_mesh_sha256": "c" * 64,
        "run_hash": "d" * 64,
        "particle": "not applicable",
        "energy_integration_range_eV": "not applicable",
        "uncertainty_meaning": "not applicable",
        "simulation_status": "direct",
        "data_status": "available",
    }


def test_complete_geometry_and_field_figure_manifest():
    validate_figure_manifest(
        {"figures": [_row(name) for name in FIGURE_FILENAMES]}
    )


def test_missing_figure_is_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        validate_figure_manifest(
            {"figures": [_row(name) for name in FIGURE_FILENAMES[:-1]]}
        )
