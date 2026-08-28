import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pytest

from parastell.reference_geometry import GEOMETRY_MODE
from parastell.reference_geometry import REFERENCE_SCHEMA
from parastell.reference_geometry import ReferenceGeometry
from parastell.reference_geometry import _outward_sign
from parastell.reference_geometry import _surface_geometry_signature
from parastell.reference_geometry import audit_volume_envelope
from parastell.reference_geometry import canonical_geometry_fingerprint
from parastell.reference_geometry import compare_reference_geometry
from parastell.reference_geometry import geometry_inventory
from parastell.reference_geometry import native_dagmc_id_inventory
from parastell.reference_geometry import sha256_file
from parastell.reference_geometry import validate_reference_manifest


TEST_FILES = Path(__file__).parent / "files_for_tests"


def _require_dagmc_python():
    # The PyPI pydagmc package may be importable while its separately supplied
    # PyMOAB runtime is absent.  Treat that host configuration as the optional
    # dependency boundary; the same tests are run without skips in the pinned
    # DAGMC/OpenMC qualification container.
    pytest.importorskip("pymoab")
    return pytest.importorskip("pydagmc")


def test_reference_geometry_reuses_original_file_byte_for_byte():
    path = TEST_FILES / "one_cube.h5m"
    before = path.read_bytes()
    reference = ReferenceGeometry.open(path, expected_sha256=sha256_file(path))

    assert reference.path == path.resolve()
    assert reference.geometry_mode == GEOMETRY_MODE
    assert reference.verify_unchanged()
    assert path.read_bytes() == before
    assert reference.evidence()["h5m_mutated"] is False


def test_reference_geometry_rejects_wrong_hash():
    with pytest.raises(ValueError, match="hash mismatch"):
        ReferenceGeometry.open(
            TEST_FILES / "one_cube.h5m", expected_sha256="0" * 64
        )


def test_same_h5m_has_byte_identical_parity():
    _require_dagmc_python()
    path = TEST_FILES / "one_cube.h5m"
    report = compare_reference_geometry(path, path)

    assert report["classification"] == "BYTE_IDENTICAL_REUSE"
    assert report["pass"] is True
    assert report["extra_physical_h5m_volumes"] == 0
    assert report["h5m_mutation_performed"] is False


def test_canonical_fingerprint_is_reproducible():
    _require_dagmc_python()
    path = TEST_FILES / "two_cubes.h5m"
    first = canonical_geometry_fingerprint(path)
    second = canonical_geometry_fingerprint(path)

    assert first["canonical_fingerprint"] == second["canonical_fingerprint"]
    assert (
        first["raw_h5m_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert first["volume_count"] == second["volume_count"]
    assert first["surface_count"] == second["surface_count"]


def test_synthetic_volume_outer_envelope_closes():
    pydagmc = _require_dagmc_python()
    path = TEST_FILES / "one_cube.h5m"
    model = pydagmc.Model(str(path))
    volume = next(iter(model.volumes_by_id.values()))
    audit = audit_volume_envelope(volume)

    assert audit["edge_closure_pass"]
    assert audit["vector_area_closure_pass"]
    assert audit["closed"]
    assert audit["edge_multiplicity_error_count"] == 0
    assert audit["directed_edge_orientation_error_count"] == 0
    assert audit["positive_signed_volume_pass"] is True


def _tetrahedron_volume(
    *, reverse_first_face=False, degenerate_first_face=False
):
    volume = SimpleNamespace(
        id=7, material="magnets", component_id="magnet-test"
    )
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    if reverse_first_face:
        faces[0] = tuple(reversed(faces[0]))
    if degenerate_first_face:
        faces[0] = (0, 0, 1)
    volume.surfaces = [
        SimpleNamespace(
            id=index + 1,
            triangle_coords=vertices[list(face)],
            forward_volume=volume,
            reverse_volume=None,
        )
        for index, face in enumerate(faces)
    ]
    return volume


def test_envelope_rejects_inconsistent_facet_winding():
    accepted = audit_volume_envelope(_tetrahedron_volume())
    rejected = audit_volume_envelope(
        _tetrahedron_volume(reverse_first_face=True)
    )

    assert accepted["closed"] is True
    assert accepted["positive_signed_volume_pass"] is True
    assert rejected["edge_multiplicity_error_count"] == 0
    assert rejected["directed_edge_orientation_error_count"] > 0
    assert rejected["closed"] is False


def test_envelope_rejects_degenerate_facets():
    rejected = audit_volume_envelope(
        _tetrahedron_volume(degenerate_first_face=True)
    )

    assert rejected["degenerate_triangle_count"] == 1
    assert rejected["nondegenerate_facets_pass"] is False
    assert rejected["closed"] is False


def test_surface_signature_preserves_winding():
    triangle = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    forward = SimpleNamespace(id=1, triangle_coords=triangle)
    cyclic = SimpleNamespace(id=2, triangle_coords=triangle[:, [1, 2, 0]])
    reverse = SimpleNamespace(id=3, triangle_coords=triangle[:, [0, 2, 1]])

    assert _surface_geometry_signature(
        forward, 1.0e-6
    ) == _surface_geometry_signature(cyclic, 1.0e-6)
    assert _surface_geometry_signature(
        forward, 1.0e-6
    ) != _surface_geometry_signature(reverse, 1.0e-6)


def test_outward_sign_rejects_same_volume_on_both_sides():
    volume = SimpleNamespace(id=7)
    surface = SimpleNamespace(
        id=9, forward_volume=volume, reverse_volume=volume
    )

    with pytest.raises(ValueError, match="both senses"):
        _outward_sign(surface, 7)


def test_native_id_inventory_controls_explicit_openmc_wrapper_ids():
    _require_dagmc_python()
    openmc = pytest.importorskip("openmc")
    path = TEST_FILES / "one_cube.h5m"
    native = native_dagmc_id_inventory(path)
    reference = ReferenceGeometry.open(path, expected_sha256=sha256_file(path))
    before = path.read_bytes()

    geometry = reference.openmc_geometry(external_vacuum_radius_cm=100.0)
    root = geometry.root_universe
    cell = next(iter(root.cells.values()))
    boundary = next(iter(cell.region.get_surfaces().values()))

    assert boundary.id == native["maximum_native_id"] + 1
    assert cell.id == native["maximum_native_id"] + 2
    assert root.id == native["maximum_native_id"] + 3
    assert cell.fill.id == native["maximum_native_id"] + 4
    assert cell.fill.auto_geom_ids is False
    assert path.read_bytes() == before


def test_geometry_inventory_has_no_synthetic_extra_volume():
    _require_dagmc_python()
    path = TEST_FILES / "one_cube.h5m"
    inventory = geometry_inventory(path)

    assert inventory["raw_h5m_sha256"] == sha256_file(path)
    assert inventory["volume_count"] == 1
    assert inventory["surface_count"] == 6


def test_reference_manifest_rejects_geometry_changes():
    manifest = {
        "schema": REFERENCE_SCHEMA,
        "reference_identity_classification": "ORIGINAL_PARASTELL_PUBLIC_EXAMPLE_NOT_MACHINE_CONFIRMED",
        "fork_main_sha": "a" * 40,
        "upstream_main_sha": "a" * 40,
        "inputs": [],
        "example_script": {},
        "environment": {},
        "raw_h5m_sha256": "b" * 64,
        "canonical_geometry_fingerprint": "c" * 64,
        "coordinate_units": "cm",
        "modeled_toroidal_extent_deg": 90.0,
        "field_period_semantics": "direct sector",
        "volume_inventory": {},
        "surface_inventory": {},
        "material_component_tags": [],
        "magnet_ids": [],
        "magnet_representation": "split_casing_and_winding",
        "source_mesh": {},
        "faceting_settings": {},
        "ports_present": False,
        "case_thickness_cm": 0.0,
    }

    with pytest.raises(ValueError, match="homogenized"):
        validate_reference_manifest(manifest)


def test_reference_manifest_requires_zero_case_thickness_and_no_ports():
    base = {
        "schema": REFERENCE_SCHEMA,
        "reference_identity_classification": "ORIGINAL_PARASTELL_PUBLIC_EXAMPLE_NOT_MACHINE_CONFIRMED",
        "fork_main_sha": "a" * 40,
        "upstream_main_sha": "a" * 40,
        "inputs": [],
        "example_script": {},
        "environment": {},
        "raw_h5m_sha256": "b" * 64,
        "canonical_geometry_fingerprint": "c" * 64,
        "coordinate_units": "cm",
        "modeled_toroidal_extent_deg": 90.0,
        "field_period_semantics": "direct sector",
        "volume_inventory": {},
        "surface_inventory": {},
        "material_component_tags": [],
        "magnet_ids": [],
        "magnet_representation": "one_homogenized_solid_each",
        "source_mesh": {},
        "faceting_settings": {},
        "ports_present": False,
        "case_thickness_cm": 0.0,
    }
    validate_reference_manifest(base)

    with pytest.raises(ValueError, match="no ports"):
        validate_reference_manifest({**base, "ports_present": True})
    with pytest.raises(ValueError, match="exactly zero"):
        validate_reference_manifest({**base, "case_thickness_cm": 1.0})
