import json
from types import SimpleNamespace

import pytest

from parastell import dagmc_envelope
from parastell import energy_groups
from parastell import magnet_openmc_model
from parastell import magnet_radiation_field
from parastell import material_manifest
from parastell.magnet_radiation_field import MagnetRadiationFieldProducer
from parastell.magnet_stage_handlers import inventory_magnets
from parastell.magnet_stage_handlers import prepare_unbiased_model
from parastell.magnet_stage_handlers import validate_geometry


POLICY = {
    "coordinate_quantum_cm": 2.0e-5,
    "faceting_tolerances": {
        "minimum_mesh_size_cm": 3.0,
        "maximum_mesh_size_cm": 12.0,
    },
}
FINGERPRINT = "a" * 64


def _association(path):
    path.write_text(
        json.dumps(
            {
                "canonical_geometry_fingerprint": FINGERPRINT,
                "canonical_geometry_policy": POLICY,
                "associations": {"16": {}},
            }
        ),
        encoding="utf-8",
    )


def test_validation_hashes_the_configured_policy(tmp_path, monkeypatch):
    geometry = tmp_path / "geometry.h5m"
    geometry.write_bytes(b"geometry")
    associations = tmp_path / "associations.json"
    _association(associations)
    captured = {}

    def fingerprint(path, **kwargs):
        captured.update(kwargs)
        return {
            "algorithm": "sha256",
            "canonical_fingerprint": FINGERPRINT,
            "raw_h5m_sha256": "b" * 64,
            "surface_count": 2,
            "volume_count": 1,
        }

    monkeypatch.setattr(
        dagmc_envelope, "canonical_dagmc_fingerprint", fingerprint
    )
    monkeypatch.setattr(
        dagmc_envelope,
        "require_watertight_dagmc",
        lambda path: SimpleNamespace(to_dict=lambda: {"status": "PASS"}),
    )
    report = validate_geometry(
        {
            "dagmc_path": str(geometry),
            "associations_path": str(associations),
            "validation_report_path": str(tmp_path / "validation.json"),
            **POLICY,
        },
        tmp_path,
    )

    assert captured == POLICY
    assert report["fingerprint"]["coordinate_quantum_cm"] == 2.0e-5
    assert (
        report["fingerprint"]["faceting_tolerances"]
        == POLICY["faceting_tolerances"]
    )


def test_inventory_inherits_policy_from_association_artifact(
    tmp_path, monkeypatch
):
    associations = tmp_path / "associations.json"
    _association(associations)
    captured = {}
    inventory = SimpleNamespace(
        canonical_geometry_fingerprint=FINGERPRINT,
        to_dict=lambda: {"inventory": "ok"},
    )

    def discover(*args, **kwargs):
        captured.update(kwargs)
        return inventory

    monkeypatch.setattr(dagmc_envelope, "discover_magnet_volumes", discover)
    monkeypatch.setattr(
        dagmc_envelope, "select_magnet_pairs", lambda *args: ()
    )
    inventory_magnets(
        {
            "dagmc_path": str(tmp_path / "geometry.h5m"),
            "associations_path": str(associations),
            "inventory_path": str(tmp_path / "inventory.json"),
        },
        tmp_path,
    )

    assert captured["coordinate_quantum_cm"] == 2.0e-5
    assert captured["faceting_tolerances"] == POLICY["faceting_tolerances"]


def test_explicit_policy_cannot_disagree_with_association_artifact(
    tmp_path, monkeypatch
):
    associations = tmp_path / "associations.json"
    _association(associations)
    monkeypatch.setattr(
        dagmc_envelope,
        "discover_magnet_volumes",
        lambda *args, **kwargs: pytest.fail("discovery must not run"),
    )

    with pytest.raises(ValueError, match="disagrees"):
        inventory_magnets(
            {
                "dagmc_path": str(tmp_path / "geometry.h5m"),
                "associations_path": str(associations),
                "inventory_path": str(tmp_path / "inventory.json"),
                "coordinate_quantum_cm": 4.0e-5,
            },
            tmp_path,
        )


def test_producer_reuses_policy_for_discovery_and_facet_ids(
    tmp_path, monkeypatch
):
    geometry = tmp_path / "geometry.h5m"
    geometry.write_bytes(b"geometry")
    captured = {"discover": None, "envelope": None}
    pair = SimpleNamespace(
        magnet_id="magnet-1",
        coil_id="coil-1",
        winding_pack=SimpleNamespace(
            volume_id=16, centroid_global_cm=(0.0, 0.0, 0.0)
        ),
    )
    inventory = SimpleNamespace(
        canonical_geometry_fingerprint=FINGERPRINT,
    )

    def discover(*args, **kwargs):
        captured["discover"] = kwargs
        return inventory

    def envelope(*args, **kwargs):
        captured["envelope"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(
        magnet_radiation_field, "discover_magnet_volumes", discover
    )
    monkeypatch.setattr(
        magnet_radiation_field,
        "select_magnet_pairs",
        lambda *args: (pair,),
    )
    monkeypatch.setattr(
        magnet_radiation_field, "extract_closed_envelope", envelope
    )
    producer = MagnetRadiationFieldProducer(
        geometry,
        **POLICY,
        expected_canonical_geometry_fingerprint=FINGERPRINT,
    )
    producer.discover()
    producer.build_envelopes(
        frames_by_magnet={
            "magnet-1": {
                "plasma_direction_global": (1.0, 0.0, 0.0),
                "toroidal_direction_global": (0.0, 1.0, 0.0),
                "poloidal_direction_global": (0.0, 0.0, 1.0),
            }
        }
    )

    for call in (captured["discover"], captured["envelope"]):
        assert call["coordinate_quantum_cm"] == 2.0e-5
        assert call["faceting_tolerances"] == POLICY["faceting_tolerances"]


def test_unbiased_model_handler_forwards_explicit_policy(
    tmp_path, monkeypatch
):
    captured = {}
    monkeypatch.setattr(
        material_manifest,
        "resolve_material_manifest",
        lambda *args, **kwargs: {"material": "resolved"},
    )
    monkeypatch.setattr(
        material_manifest,
        "audit_nuclear_data",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        energy_groups,
        "get_structure",
        lambda *args, **kwargs: SimpleNamespace(edges_eV=(0.0, 1.0)),
    )

    def prepare(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "PREPARED"}

    monkeypatch.setattr(
        magnet_openmc_model, "prepare_magnet_openmc_model", prepare
    )
    result = prepare_unbiased_model(
        {
            "material_config": "materials.json",
            "resolved_material_manifest_path": str(
                tmp_path / "resolved-materials.json"
            ),
            "cross_sections_path": "cross_sections.xml",
            "approved_library": "library",
            "evaluation_release": "release",
            "nuclear_data_audit_path": str(tmp_path / "nuclear.json"),
            "model_directory": str(tmp_path / "model"),
            "dagmc_path": "geometry.h5m",
            "source_mesh_path": "source.h5m",
            "associations_path": "associations.json",
            "particles_per_batch": 10,
            "batches": 2,
            "seed": 3,
            "max_surface_particles": 100,
            **POLICY,
        },
        tmp_path,
    )

    assert result == {"status": "PREPARED"}
    assert captured["coordinate_quantum_cm"] == 2.0e-5
    assert captured["faceting_tolerances"] == POLICY["faceting_tolerances"]
