import h5py
import numpy as np
import pytest

from parastell.dagmc_envelope import DagmcEnvelope
from parastell.dagmc_envelope import FacetedSurface
from parastell.magnet_boundary_envelope import EnvelopeSurface
from parastell.magnet_boundary_envelope import MagnetBoundaryEnvelope
from parastell.magnet_boundary_envelope import build_correlated_bank
from parastell.magnet_boundary_envelope import read_handoff
from parastell.magnet_boundary_envelope import write_handoff


def _normalization():
    return {
        "basis": "per source history",
        "particles_per_source_history": 1.0,
        "area_basis": "surface integrated",
        "energy_bin_width": "group integrated",
        "solid_angle_measure": "angle-bin integrated",
        "quantity": "partial crossing current",
        "time_basis": "none for per-source values",
    }


def test_v22_handoff_persists_complete_triangle_and_record_mapping(tmp_path):
    triangle = np.asarray(
        [[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]
    )
    faceted = FacetedSurface(
        29,
        "plasma_facing",
        triangle,
        np.asarray([[1.0, 0.0, 0.0]]),
        np.asarray([0.5]),
        np.asarray([1.0, 1.0 / 3.0, 1.0 / 3.0]),
        ("facet-stable",),
        8,
    )
    surface = EnvelopeSurface(
        29,
        "plasma_facing",
        0.5,
        (1.0, 1.0 / 3.0, 1.0 / 3.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        vector_area_global_cm2=(0.0, 0.0, 0.0),
    )
    envelope = MagnetBoundaryEnvelope(
        "winding-pack-8",
        "magnet-8",
        8,
        (surface,),
        "a" * 64,
        metadata={"canonical_geometry_fingerprint": "c" * 64},
    )
    dagmc = DagmcEnvelope(envelope, (faceted,), 3, 0, 0.0)
    mapping = dagmc.facet_mappings(
        [29], [[1.0 + 1.0e-8, 0.25, 0.25]], require_valid=True
    )
    normals = mapping.pop("outward_normal_global")
    bank = build_correlated_bank(
        envelope,
        position_global_cm=np.asarray([[1.0 + 1.0e-8, 0.25, 0.25]]),
        direction_global=np.asarray([[-1.0, 0.0, 0.0]]),
        energy_eV=[14.1e6],
        raw_weight=[1.0],
        openmc_weight=[100.0],
        particle=["neutron"],
        surface_id=[29],
        time_s=[2.0e-9],
        delayed_group=[0],
        energy_edges_eV=[0.0, 20.0e6],
        outward_normal_global=normals,
        facet_mapping=mapping,
    )
    output = tmp_path / "facet-complete-v22.h5"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        write_handoff(
            tmp_path / "mismatch.h5",
            envelope,
            bank,
            provenance={
                "openmc_version": "test",
                "parastell_commit": "test",
                "canonical_geometry_fingerprint": "d" * 64,
            },
            normalization=_normalization(),
            facet_catalog=dagmc.facet_metadata(),
        )
    manifest = write_handoff(
        output,
        envelope,
        bank,
        provenance={
            "openmc_version": "test",
            "parastell_commit": "test",
            "canonical_geometry_fingerprint": "c" * 64,
        },
        normalization=_normalization(),
        facet_catalog=dagmc.facet_metadata(),
    )

    assert manifest["schema"] == "parastell.magnet_boundary_source/v2.2.0"
    assert manifest["facet_catalog"]["facet_complete_v22"]
    assert manifest["facet_complete_boundary"] == {
        "schema_revision": "parastell.magnet_boundary_source/v2.2.0",
        "record_fields_complete": True,
        "all_records_valid": True,
        "catalog_complete": True,
    }
    with h5py.File(output) as handle:
        assert handle["facet_catalog/triangle_vertices_global_cm"].shape == (
            1,
            3,
            3,
        )
        assert (
            handle["records/canonical_facet_id"].asstr()[0] == "facet-stable"
        )
        assert handle["records/openmc_weight"][0] == 100.0
        assert handle["records/delayed_group"][0] == 0
    restored_manifest, restored_envelope, restored = read_handoff(output)
    assert restored_manifest["schema_version"] == "2.2.0"
    assert restored_envelope.dagmc_volume_id == 8
    assert restored.columns["inside_facet"].tolist() == [True]
