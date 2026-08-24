import argparse
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from parastell.dagmc_envelope import canonical_dagmc_fingerprint
from parastell.magnet_radiation_field_bundle import read_radiation_field_bundle
from parastell.magnet_radiation_field_bundle import (
    write_radiation_field_bundle,
)
from parastell import magnet_handoff_cli


def test_canonical_fingerprint_ignores_triangle_and_entity_order(
    tmp_path, monkeypatch
):
    path = tmp_path / "geometry.h5m"
    path.write_bytes(b"raw bytes may differ")
    first = SimpleNamespace(id=1, material="winding_pack", name="coil-0")
    outside = SimpleNamespace(id=2, material="vacuum", name="outside")
    triangles = np.asarray(
        [[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 1]]],
        dtype=float,
    )
    surface = SimpleNamespace(
        id=7,
        triangle_coords=triangles.reshape(-1, 3),
        forward_volume=outside,
        reverse_volume=first,
    )
    first.surfaces = [surface]
    outside.surfaces = [surface]
    module = SimpleNamespace(
        Model=lambda value: SimpleNamespace(
            volumes_by_id={2: outside, 1: first}
        )
    )
    monkeypatch.setitem(sys.modules, "pydagmc", module)

    original = canonical_dagmc_fingerprint(path)
    surface.triangle_coords = triangles[::-1, ::-1].reshape(-1, 3)
    reordered = canonical_dagmc_fingerprint(path)

    assert (
        original["canonical_fingerprint"] == reordered["canonical_fingerprint"]
    )
    assert original["raw_h5m_sha256"] == reordered["raw_h5m_sha256"]


def test_neutral_bundle_round_trip_and_surface_current_rejection(tmp_path):
    boundary = tmp_path / "boundary.h5"
    flux = tmp_path / "flux.h5"
    boundary.write_bytes(b"boundary")
    flux.write_bytes(b"scalar flux")
    bundle = tmp_path / "bundle"
    result = write_radiation_field_bundle(
        bundle,
        provenance={"parastell_commit": "abc", "openmc_version": "0.16.0"},
        geometry={
            "raw_h5m_sha256": "0" * 64,
            "canonical_geometry_fingerprint": "1" * 64,
        },
        source={
            "physical_source_rate_per_s": 2.4e20,
            "source_definition_sha256": "2" * 64,
        },
        nuclear_data={"library": "qualified-test"},
        materials={"manifest_sha256": "3" * 64},
        magnet_inventory=[
            {"magnet_id": "coil-0", "winding_pack_volume_id": 16}
        ],
        products=[
            {
                "kind": "boundary_phase_space",
                "magnet_id": "coil-0",
                "path": boundary,
                "units": "1/source",
                "normalization": "per_source_history",
            },
            {
                "kind": "volume_scalar_flux",
                "quantity": "volume_scalar_flux",
                "magnet_id": "coil-0",
                "path": flux,
                "units": "particles/cm2/s",
                "normalization": "physical_source_rate",
            },
        ],
        verification={"same_run_integrity": "pass"},
    )
    restored = read_radiation_field_bundle(bundle)
    assert restored["schema"] == result["schema"]
    assert len(restored["products"]) == 2
    manifest_path = bundle / "manifest.json"
    malformed = json.loads(manifest_path.read_text())
    malformed["products"][1]["quantity"] = "surface_current"
    manifest_path.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="surface current"):
        read_radiation_field_bundle(bundle)


def test_bundle_cli_resolves_products_and_binds_geometry(
    tmp_path, monkeypatch
):
    product = tmp_path / "volume.h5"
    product.write_bytes(b"volume")
    specification = {
        "provenance": {"parastell_commit": "abc"},
        "source": {
            "source_definition_sha256": "source-hash",
            "physical_source_rate_per_s": 5.0,
        },
        "nuclear_data": {"library": "fixture"},
        "materials": {"M1": "fixture"},
        "magnet_inventory": [{"magnet_id": "M1", "dagmc_volume_id": 1}],
        "products": [
            {
                "kind": "volume_scalar_flux",
                "magnet_id": "M1",
                "path": product.name,
                "units": "particles/cm2/s",
                "normalization": "physical",
                "quantity": "volume_scalar_flux",
            }
        ],
        "verification": {"test": True},
    }
    spec_path = tmp_path / "bundle.json"
    spec_path.write_text(json.dumps(specification), encoding="utf-8")
    monkeypatch.setattr(
        magnet_handoff_cli,
        "canonical_dagmc_fingerprint",
        lambda _path: {
            "canonical_fingerprint": "canonical-hash",
            "raw_h5m_sha256": "raw-hash",
        },
    )
    result = magnet_handoff_cli._bundle(
        argparse.Namespace(
            spec=spec_path,
            dagmc=tmp_path / "geometry.h5m",
            output_dir=tmp_path / "bundle",
        )
    )
    assert (
        result["geometry"]["canonical_geometry_fingerprint"]
        == "canonical-hash"
    )
    assert (
        result["products"][0]["sha256"]
        == hashlib.sha256(b"volume").hexdigest()
    )
