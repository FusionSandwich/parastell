import numpy as np
import pytest

from parastell.dagmc_envelope import FacetedSurface
from parastell.dagmc_envelope import _canonical_facet_id
from parastell.magnet_boundary_envelope import EnvelopeSurface
from parastell.magnet_boundary_envelope import MagnetBoundaryEnvelope
from parastell.magnet_boundary_envelope import build_correlated_bank


def _surface(surface_id, centroid, normal, tangent, width):
    return EnvelopeSurface(
        surface_id=surface_id,
        role="outer",
        area_cm2=4.0,
        centroid_global_cm=centroid,
        outward_normal_global=normal,
        toroidal_direction_global=tangent,
        poloidal_direction_global=width,
        u_edges_cm=(-1.0, 0.0, 1.0),
        v_edges_cm=(-1.0, 0.0, 1.0),
    )


def _box():
    return MagnetBoundaryEnvelope(
        envelope_id="outer-magnet-A",
        magnet_component="magnet-A",
        dagmc_volume_id=7,
        dagmc_geometry_sha256="a" * 64,
        surfaces=(
            _surface(1, (1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            _surface(2, (-1, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1)),
            _surface(3, (0, 1, 0), (0, 1, 0), (-1, 0, 0), (0, 0, 1)),
            _surface(4, (0, -1, 0), (0, -1, 0), (1, 0, 0), (0, 0, 1)),
            _surface(5, (0, 0, 1), (0, 0, 1), (1, 0, 0), (0, 1, 0)),
            _surface(6, (0, 0, -1), (0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        ),
    )


def test_canonical_facet_id_is_stable_under_triangle_start_vertex_reordering():
    triangle = np.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    expected = _canonical_facet_id("b" * 64, 1, triangle, 1, 1.0e-6)
    for shift in (1, 2):
        assert (
            _canonical_facet_id(
                "b" * 64, 1, np.roll(triangle, shift, axis=0), 1, 1.0e-6
            )
            == expected
        )


def test_facet_mapping_preserves_barycentrics_and_nearest_status():
    triangle = np.asarray(
        [[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]
    )
    surface = FacetedSurface(
        surface_id=1,
        role="outer",
        triangles_cm=triangle,
        triangle_normals_outward=np.asarray([[1.0, 0.0, 0.0]]),
        triangle_areas_cm2=np.asarray([0.5]),
        centroid_global_cm=np.asarray([1.0, 1.0 / 3.0, 1.0 / 3.0]),
        canonical_facet_ids=("facet-stable",),
    )
    exact = surface.locate([1.0, 0.2, 0.3])
    assert exact["facet_id"] == "facet-stable"
    assert exact["mapping_status"] == "CONTAINING_FACET"
    assert exact["barycentric_coordinates"] == pytest.approx([0.5, 0.2, 0.3])
    reconstructed = exact["barycentric_coordinates"] @ triangle[0]
    assert reconstructed == pytest.approx([1.0, 0.2, 0.3])

    nearest = surface.locate([1.01, 2.0, 2.0])
    assert nearest["mapping_status"] == "NEAREST_FACET"
    assert nearest["distance_to_facet_residual_cm"] == pytest.approx(0.01)


def test_boundary_record_keeps_direction_energy_time_and_facet_fields_correlated():
    bank = build_correlated_bank(
        _box(),
        position_global_cm=np.asarray([[1.0, 0.2, 0.3]]),
        direction_global=np.asarray([[-1.0, 0.0, 0.0]]),
        energy_eV=[14.1e6],
        raw_weight=[0.25],
        particle=["neutron"],
        surface_id=[1],
        time_s=[2.5e-8],
        energy_edges_by_particle={
            "neutron": [0.0, 14.0e6, 14.2e6, 20.0e6],
            "photon": [0.0, 1.0e9],
        },
        facet_mapping={
            "facet_id": ["facet-stable"],
            "facet_index": [0],
            "barycentric_coordinates": [[0.5, 0.2, 0.3]],
            "distance_to_facet_residual_cm": [0.0],
            "facet_mapping_status": ["CONTAINING_FACET"],
        },
    )
    assert bank.columns["position_local_cm"][0] == pytest.approx(
        [0.2, 0.3, 0.0]
    )
    assert bank.columns["direction_local"][0] == pytest.approx(
        [0.0, 0.0, -1.0]
    )
    assert bank.columns["crossing_sense"].tolist() == ["incoming"]
    assert bank.columns["energy_eV"].tolist() == [14.1e6]
    assert bank.columns["energy_group"].tolist() == [1]
    assert bank.columns["time_s"].tolist() == [2.5e-8]
    time_status = bank.metadata["field_availability"]["time_s"]
    assert time_status["available"] is True
    assert time_status["semantics"] == "prompt_particle_flight_time"
    assert bank.columns["facet_id"].tolist() == ["facet-stable"]
