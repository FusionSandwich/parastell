from types import SimpleNamespace

import numpy as np

from parastell.magnet_surface_audit import classify_casing_surfaces


def _surface(surface_id, forward, reverse):
    return SimpleNamespace(
        id=surface_id,
        forward_volume=forward,
        reverse_volume=reverse,
        triangle_coords=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
    )


def test_external_casing_selection_uses_only_accepted_adjacent_regions():
    casing = SimpleNamespace(id=7)
    winding = SimpleNamespace(id=8)
    vacuum = SimpleNamespace(id=43)
    coolant = SimpleNamespace(id=45)
    casing.surfaces = (
        _surface(10, casing, vacuum),
        _surface(11, casing, winding),
        _surface(12, casing, coolant),
    )
    winding.surfaces = (casing.surfaces[1],)
    result = classify_casing_surfaces(
        casing, winding, accepted_external_volume_ids=(43,)
    )
    assert result["outer_casing_external_surface_ids"] == [10]
    assert result["casing_internal_surface_ids"] == [11]
    assert result["casing_other_excluded_surface_ids"] == [12]
    assert not result["coordinate_sign_heuristic_used"]
