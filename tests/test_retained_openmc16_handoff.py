import hashlib
import json

import h5py
import numpy as np
import pytest

from parastell.coil_frame import parallel_transport_frame
from parastell.retained_openmc16_handoff import (
    load_handoff_manifest,
    reconstruct_hash_bound_frame,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_handoff_manifest_reads_only_embedded_contract(tmp_path):
    path = tmp_path / "handoff.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "manifest_json", data=json.dumps({"schema": "x"})
        )
    assert load_handoff_manifest(path) == {"schema": "x"}


def test_reconstruct_hash_bound_frame_rejects_wrong_h5m(tmp_path):
    h5m = tmp_path / "geometry.h5m"
    h5m.write_bytes(b"geometry")
    other = tmp_path / "other.h5m"
    other.write_bytes(b"other")
    points = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    frame = parallel_transport_frame(points, closed=True)
    samples = [
        {
            "tangent_global": tangent.tolist(),
            "width_direction_global": width.tolist(),
            "normal_direction_global": normal.tolist(),
        }
        for tangent, width, normal in zip(
            frame.tangents,
            frame.radial_directions,
            frame.transverse_directions,
        )
    ]
    recipe = {
        "schema": "parastell.magnet_geometry_interchange/v1.0.0",
        "geometry_fingerprint": "f" * 64,
        "magnets": [
            {
                "magnet_id": "magnet-8",
                "artifacts": {
                    "outer_h5m": {
                        "status": "available",
                        "sha256": _sha256(h5m),
                    }
                },
                "centreline": {"points_global_cm": points.tolist()},
                "closure": {"represented_segment_closed": True},
                "frame": {
                    "samples": samples,
                    "closure_twist_rad": frame.closure_twist_rad,
                },
            }
        ],
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="H5M hash"):
        reconstruct_hash_bound_frame(
            recipe_path,
            other,
            magnet_id="magnet-8",
            expected_geometry_fingerprint="f" * 64,
        )
    _, magnet, restored = reconstruct_hash_bound_frame(
        recipe_path,
        h5m,
        magnet_id="magnet-8",
        expected_geometry_fingerprint="f" * 64,
    )
    assert magnet["magnet_id"] == "magnet-8"
    assert np.allclose(restored.tangents, frame.tangents)
