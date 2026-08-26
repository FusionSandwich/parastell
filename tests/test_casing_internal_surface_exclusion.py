from types import SimpleNamespace

import numpy as np

from parastell.magnet_surface_audit import classify_casing_surfaces


def test_shared_casing_winding_face_is_never_an_external_face():
    casing = SimpleNamespace(id=1)
    winding = SimpleNamespace(id=2)
    shared = SimpleNamespace(
        id=99,
        forward_volume=casing,
        reverse_volume=winding,
        triangle_coords=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
    )
    casing.surfaces = (shared,)
    winding.surfaces = (shared,)
    result = classify_casing_surfaces(
        casing, winding, accepted_external_volume_ids=(2, 43)
    )
    assert result["outer_casing_external_surface_ids"] == []
    assert result["casing_internal_surface_ids"] == [99]
