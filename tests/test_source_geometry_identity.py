from pathlib import Path

import numpy as np

from parastell.source_geometry_identity import canonical_source_mesh_arrays
from parastell.source_geometry_identity import canonical_step_payload_sha256


def test_step_identity_ignores_only_export_timestamp(tmp_path: Path):
    first = tmp_path / "first.step"
    second = tmp_path / "second.step"
    template = """ISO-10303-21;
HEADER;
FILE_NAME('Open CASCADE Shape Model','{timestamp}',('Author'),('Open CASCADE'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(1.,2.,3.));
ENDSEC;
END-ISO-10303-21;
"""
    first.write_text(template.format(timestamp="2026-08-26T21:13:42"))
    second.write_text(template.format(timestamp="2026-08-27T05:10:23"))
    assert canonical_step_payload_sha256(
        first
    ) == canonical_step_payload_sha256(second)
    second.write_text(
        template.format(timestamp="2026-08-27T05:10:23").replace(
            "(1.,2.,3.)", "(1.,2.,4.)"
        )
    )
    assert canonical_step_payload_sha256(
        first
    ) != canonical_step_payload_sha256(second)


def test_source_mesh_identity_ignores_handle_and_orientation_order():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
        ]
    )
    tetrahedra = np.array(
        [
            vertices[[0, 1, 2, 3]],
            vertices[[1, 2, 3, 4]],
        ]
    )
    first = canonical_source_mesh_arrays(
        vertices, tetrahedra, [10.0, 20.0], [1.0 / 6.0, 0.5]
    )
    second = canonical_source_mesh_arrays(
        vertices[::-1],
        tetrahedra[::-1, ::-1],
        [20.0, 10.0],
        [0.5, 1.0 / 6.0],
    )
    assert first["canonical_fingerprint"] == second["canonical_fingerprint"]


def test_source_mesh_identity_covers_strengths():
    vertices = np.eye(3)
    vertices = np.vstack(([0.0, 0.0, 0.0], vertices))
    tetrahedra = vertices.reshape((1, 4, 3))
    first = canonical_source_mesh_arrays(vertices, tetrahedra, [10.0], [1.0])
    second = canonical_source_mesh_arrays(vertices, tetrahedra, [11.0], [1.0])
    assert first["canonical_fingerprint"] != second["canonical_fingerprint"]
