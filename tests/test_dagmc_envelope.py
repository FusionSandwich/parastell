from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from parastell.dagmc_envelope import extract_closed_envelope
from parastell.dagmc_envelope import _openmc_to_outward_normal_sign


def test_openmc_normal_sign_uses_target_volume_sense():
    volume = SimpleNamespace(id=16)
    outside = SimpleNamespace(id=15)
    reverse = SimpleNamespace(
        id=73, forward_volume=outside, reverse_volume=volume
    )
    forward = SimpleNamespace(
        id=82, forward_volume=volume, reverse_volume=None
    )

    assert _openmc_to_outward_normal_sign(reverse, volume.id) == 1
    assert _openmc_to_outward_normal_sign(forward, volume.id) == -1

    missing = SimpleNamespace(
        id=99, forward_volume=outside, reverse_volume=None
    )
    with pytest.raises(ValueError, match="no sense"):
        _openmc_to_outward_normal_sign(missing, volume.id)


def test_real_magnet_volume_is_closed_when_asset_available():
    path = Path(
        "validation_output/continuation_final/magnet_retry/parastell_magnet.h5m"
    )
    if not path.is_file():
        pytest.skip("real generated magnet DAGMC is unavailable")
    result = extract_closed_envelope(
        path,
        1,
        envelope_id="magnet-1-envelope",
        magnet_id="magnet-1",
        plasma_direction_global=(-1.0, 0.0, 0.0),
        toroidal_direction_global=(0.0, 1.0, 0.0),
        poloidal_direction_global=(0.0, 0.0, 1.0),
    )
    assert result.envelope.watertight
    assert set(result.envelope.surface_ids) == set(range(1, 9))
    assert result.maximum_edge_multiplicity_error == 0
    assert result.vector_area_closure_relative < 1.0e-7
    point = result.faceted_surfaces[0].triangles_cm[0].mean(axis=0)
    assert np.linalg.norm(
        result.faceted_surfaces[0].normal_at(point)
    ) == pytest.approx(1.0)
