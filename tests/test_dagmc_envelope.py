from pathlib import Path

import numpy as np
import pytest

from parastell.dagmc_envelope import extract_closed_envelope


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
