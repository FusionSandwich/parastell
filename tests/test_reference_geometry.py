import hashlib
from pathlib import Path

import pytest

from parastell.reference_geometry import GEOMETRY_MODE
from parastell.reference_geometry import REFERENCE_SCHEMA
from parastell.reference_geometry import ReferenceGeometry
from parastell.reference_geometry import audit_volume_envelope
from parastell.reference_geometry import canonical_geometry_fingerprint
from parastell.reference_geometry import compare_reference_geometry
from parastell.reference_geometry import geometry_inventory
from parastell.reference_geometry import sha256_file
from parastell.reference_geometry import validate_reference_manifest


TEST_FILES = Path(__file__).parent / "files_for_tests"


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
    path = TEST_FILES / "one_cube.h5m"
    report = compare_reference_geometry(path, path)

    assert report["classification"] == "BYTE_IDENTICAL_REUSE"
    assert report["pass"] is True
    assert report["extra_physical_h5m_volumes"] == 0
    assert report["h5m_mutation_performed"] is False


def test_canonical_fingerprint_is_reproducible():
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
    pydagmc = pytest.importorskip("pydagmc")
    path = TEST_FILES / "one_cube.h5m"
    model = pydagmc.Model(str(path))
    volume = next(iter(model.volumes_by_id.values()))
    audit = audit_volume_envelope(volume)

    assert audit["edge_closure_pass"]
    assert audit["vector_area_closure_pass"]
    assert audit["closed"]
    assert audit["edge_multiplicity_error_count"] == 0


def test_geometry_inventory_has_no_synthetic_extra_volume():
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
