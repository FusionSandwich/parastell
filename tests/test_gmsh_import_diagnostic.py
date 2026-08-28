import copy

import pytest

from parastell.gmsh_import_diagnostic import volume_signature_delta


def _signature():
    return {
        "mass_cm3": 8.0,
        "center_cm": [1.0, 2.0, 3.0],
        "bounding_box_cm": [0.0, 1.0, 2.0, 2.0, 3.0, 4.0],
        "inertia_cm5": [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0],
    }


def test_volume_signature_delta_preserves_scale_evidence():
    source = _signature()
    imported = copy.deepcopy(source)
    imported["mass_cm3"] = 8000.0
    imported["center_cm"] = [10.0, 20.0, 30.0]
    imported["bounding_box_cm"] = [0.0, 10.0, 20.0, 20.0, 30.0, 40.0]

    result = volume_signature_delta(source, imported)

    assert result["mass_ratio_imported_over_source"] == pytest.approx(1000.0)
    assert result["linear_scale_from_mass_ratio"] == pytest.approx(10.0)
    assert result["maximum_absolute_center_delta_cm"] == 27.0
    assert result["maximum_absolute_bounding_box_delta_cm"] == 36.0
    assert result["existing_strict_match"] is False


def test_volume_signature_delta_rejects_nonpositive_mass():
    imported = _signature()
    imported["mass_cm3"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        volume_signature_delta(_signature(), imported)
