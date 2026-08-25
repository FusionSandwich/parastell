from pathlib import Path
from types import SimpleNamespace

import numpy as np

from parastell.dagmc_envelope import canonical_dagmc_fingerprint


QUANTUM_CM = 1.0e-3
VERTICES = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
)
BASE_TRIANGLES = ((0, 1, 2), (0, 2, 3))


def _model(
    *,
    triangles=BASE_TRIANGLES,
    vertex_offset=None,
    material="winding_pack",
    name="component-coil-0001",
    component_id="",
    swap_sense=False,
):
    vertices = VERTICES.copy()
    if vertex_offset is not None:
        vertex_id, coordinate, offset = vertex_offset
        vertices[vertex_id, coordinate] += offset
    triangle_coordinates = np.asarray(
        [[vertices[index] for index in triangle] for triangle in triangles]
    )
    inside = SimpleNamespace(id=10)
    outside = SimpleNamespace(id=0)
    surface = SimpleNamespace(
        id=20,
        triangle_coords=triangle_coordinates.reshape((-1, 3)),
        forward_volume=inside if swap_sense else outside,
        reverse_volume=outside if swap_sense else inside,
    )
    volume = SimpleNamespace(
        id=10,
        name=name,
        component_id=component_id,
        material=material,
        surfaces=[surface],
    )
    return SimpleNamespace(volumes_by_id={10: volume})


def _fingerprint(monkeypatch, tmp_path, models):
    paths = []
    registry = {}
    for index, model in enumerate(models):
        path = tmp_path / f"geometry-{index}.h5m"
        path.write_bytes(f"raw-build-{index}".encode())
        paths.append(path)
        registry[str(path.resolve())] = model
    monkeypatch.setitem(
        __import__("sys").modules,
        "pydagmc",
        SimpleNamespace(
            Model=lambda path: registry[str(Path(path).resolve())]
        ),
    )
    return [
        canonical_dagmc_fingerprint(
            path,
            coordinate_quantum_cm=QUANTUM_CM,
            faceting_tolerances={"deviation_distance_cm": 0.01},
        )
        for path in paths
    ]


def test_equivalent_reordered_geometry_ignores_sub_quantum_noise(
    monkeypatch, tmp_path
):
    noisy_vertices = VERTICES + np.asarray([2.0e-4, -2.0e-4, 1.0e-4])
    reordered = ((2, 3, 0), (1, 2, 0))
    noisy_model = _model(triangles=reordered)
    noisy_model.volumes_by_id[10].surfaces[0].triangle_coords = np.asarray(
        [
            [noisy_vertices[index] for index in triangle]
            for triangle in reordered
        ]
    ).reshape((-1, 3))

    reference, equivalent = _fingerprint(
        monkeypatch, tmp_path, [_model(), noisy_model]
    )

    assert reference["raw_h5m_sha256"] != equivalent["raw_h5m_sha256"]
    assert (
        reference["canonical_fingerprint"]
        == equivalent["canonical_fingerprint"]
    )
    assert reference["canonical_geometry"] == equivalent["canonical_geometry"]
    assert reference["canonical_geometry"]["canonicalization_version"] == 2
    assert reference["canonical_geometry"]["quantized_vertices"]
    assert reference["canonical_geometry"]["surfaces"][0][
        "canonical_triangle_connectivity"
    ]


def test_above_quantum_coordinate_mutation_changes_identity(
    monkeypatch, tmp_path
):
    reference, moved = _fingerprint(
        monkeypatch,
        tmp_path,
        [_model(), _model(vertex_offset=(1, 0, 2.0 * QUANTUM_CM))],
    )

    assert reference["canonical_fingerprint"] != moved["canonical_fingerprint"]


def test_surface_sense_mutation_changes_identity(monkeypatch, tmp_path):
    reference, reversed_sense = _fingerprint(
        monkeypatch, tmp_path, [_model(), _model(swap_sense=True)]
    )

    assert (
        reference["canonical_fingerprint"]
        != reversed_sense["canonical_fingerprint"]
    )


def test_material_and_component_mutations_change_identity(
    monkeypatch, tmp_path
):
    reference, material, component_name, component_id = _fingerprint(
        monkeypatch,
        tmp_path,
        [
            _model(),
            _model(material="magnet_casing"),
            _model(name="component-coil-0002"),
            _model(component_id="stable-component-identity"),
        ],
    )

    identities = {
        item["canonical_fingerprint"]
        for item in (reference, material, component_name, component_id)
    }
    assert len(identities) == 4


def test_triangle_topology_mutation_changes_identity(monkeypatch, tmp_path):
    alternate_diagonal = ((0, 1, 3), (1, 2, 3))
    reference, remeshed = _fingerprint(
        monkeypatch, tmp_path, [_model(), _model(triangles=alternate_diagonal)]
    )

    assert (
        reference["canonical_geometry"]["quantized_vertices"]
        == remeshed["canonical_geometry"]["quantized_vertices"]
    )
    assert (
        reference["canonical_geometry"]["surfaces"][0]["area_cm2"]
        == remeshed["canonical_geometry"]["surfaces"][0]["area_cm2"]
    )
    assert (
        reference["canonical_fingerprint"] != remeshed["canonical_fingerprint"]
    )
