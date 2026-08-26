import hashlib
import struct

import numpy as np
import pytest

from parastell.magnet_geometry_artifacts import write_binary_stl


def _tetrahedron_triangles():
    return np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )


def test_binary_stl_is_deterministic_under_triangle_and_start_reordering(
    tmp_path,
):
    first = _tetrahedron_triangles()
    second = first[::-1].copy()
    second[::2] = np.roll(second[::2], 1, axis=1)
    one = tmp_path / "one.stl"
    two = tmp_path / "two.stl"
    first_manifest = write_binary_stl(one, first)
    second_manifest = write_binary_stl(two, second)
    assert one.read_bytes() == two.read_bytes()
    assert first_manifest["sha256"] == second_manifest["sha256"]
    assert (
        first_manifest["sha256"]
        == hashlib.sha256(one.read_bytes()).hexdigest()
    )
    assert first_manifest["triangle_count"] == 4
    assert first_manifest["size_bytes"] == 84 + 50 * 4
    assert struct.unpack("<I", one.read_bytes()[80:84]) == (4,)
    assert first_manifest["coordinate_units"] == "cm"


def test_binary_stl_refuses_overwrite_and_degenerate_triangles(tmp_path):
    output = tmp_path / "surface.stl"
    write_binary_stl(output, _tetrahedron_triangles())
    with pytest.raises(FileExistsError):
        write_binary_stl(output, _tetrahedron_triangles())
    with pytest.raises(ValueError, match="degenerate"):
        write_binary_stl(
            tmp_path / "bad.stl",
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        )
