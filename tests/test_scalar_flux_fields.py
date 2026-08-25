import h5py
import numpy as np
import pytest

from parastell.energy_groups import get_structure
from parastell.magnet_volume_flux import export_scalar_flux_fields
from parastell.magnet_volume_flux import qualify_flux_statistics
from parastell.magnet_volume_flux import validate_spectra_pka_ready_flux


def test_ccfe_scalar_flux_field_has_exact_physical_normalization(tmp_path):
    edges = get_structure("CCFE-709", particle="neutron").edges_eV
    track = np.full((1, 709), 4.0)
    output = tmp_path / "flux.h5"
    export_scalar_flux_fields(
        output,
        fields=[
            {
                "name": "neutron_ccfe_709",
                "field_kind": "whole_volume",
                "particle": "neutron",
                "energy_edges_eV": edges,
                "energy_structure": "CCFE-709",
                "track_length_mean_cm_per_source": track,
                "track_length_std_dev_cm_per_source": np.full((1, 709), 0.4),
                "volume_cm3": [2.0],
                "region_ids": ["cell-16"],
                "magnet_ids": ["magnet-A"],
                "global_centroid_cm": [[1.0, 2.0, 3.0]],
                "local_centreline_coordinates_cm": [[4.0, 5.0, 6.0]],
                "nearest_centreline_global_cm": [[0.0, 0.0, 0.0]],
                "centreline_arclength_cm": [4.0],
                "normalized_arclength": [0.25],
                "centreline_tangent": [[1.0, 0.0, 0.0]],
                "centreline_radial": [[0.0, 1.0, 0.0]],
                "centreline_transverse": [[0.0, 0.0, 1.0]],
                "distance_to_centreline_cm": [5.0],
                "centreline_linkage_status": [
                    "LINKED_NEAREST_CENTRELINE_SEGMENT"
                ],
                "frame_type": ["coil_centerline_parallel_transport"],
                "frame_quality_status": ["PASS"],
                "material_ids": ["WindingPackReference"],
            }
        ],
        physical_source_rate_per_s=3.0,
        provenance={
            "raw_h5m_sha256": "1" * 64,
            "canonical_geometry_fingerprint": "2" * 64,
            "source_definition_sha256": "3" * 64,
            "source_mesh_sha256": "4" * 64,
            "nuclear_data_manifest_sha256": "5" * 64,
        },
        material_manifest_sha256="6" * 64,
    )
    result = validate_spectra_pka_ready_flux(
        output, field_name="neutron_ccfe_709"
    )
    assert result["groups"] == 709
    with h5py.File(output, "r+") as source:
        field = source["scalar_flux_fields/neutron_ccfe_709"]
        assert np.all(field["mean_per_source"][...] == 2.0)
        assert np.all(field["mean_physical"][...] == 6.0)
        assert field.attrs["centreline_linkage_available"]
        assert field["centreline_arclength_cm"][...].tolist() == [4.0]
        assert field["frame_type"].asstr()[...].tolist() == [
            "coil_centerline_parallel_transport"
        ]
        source.attrs["quantity"] = "surface_current"
    with pytest.raises(ValueError, match="not surface current"):
        validate_spectra_pka_ready_flux(output, field_name="neutron_ccfe_709")


def test_empty_flux_bin_is_not_claimed_as_physical_zero():
    result = qualify_flux_statistics(
        raw_count=0,
        sum_weights=0.0,
        sum_squared_weights=0.0,
        mean=0.0,
        standard_error=0.0,
    )
    assert result["status"] == "EMPTY"
    assert result["empty_bin_is_physical_zero"] is False
    assert result["zero_count_upper_bound_expected_events"] > 0.0
