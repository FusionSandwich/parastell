import pytest

from parastell.surface_source_instrumentation import (
    build_surface_instrumentation_spec,
)


def test_input_general_spec_preserves_ids_axes_and_mpi_capacity():
    spec = build_surface_instrumentation_spec(
        surface_ids=[91, 12, 47],
        energy_edges_by_particle={
            "neutron": [0.0, 1.0e6, 20.0e6],
            "photon": [0.0, 1.0e5, 20.0e6],
        },
        openmc_normal_sign_by_surface={91: 1, 12: -1, 47: 1},
        max_particles_per_process=1000,
        max_source_files=3,
        mpi_ranks=8,
    )
    assert spec["surface_ids"] == [12, 47, 91]
    assert spec["particles"] == ["neutron", "photon"]
    assert spec["native_mu_edges"] == [-1.0, 0.0, 1.0]
    assert spec["openmc_normal_sign_by_surface"] == {
        "12": -1,
        "47": 1,
        "91": 1,
    }
    assert spec["per_file_capacity"] == 8000
    assert spec["configured_capacity"] == 24_000
    assert spec["records_both_directions"]


@pytest.mark.parametrize(
    "updates, match",
    [
        ({"surface_ids": [1, 1]}, "surface_ids"),
        ({"surface_ids": []}, "surface_ids"),
        ({"energy_edges_by_particle": {"electron": [0, 1]}}, "unsupported"),
        ({"energy_edges_by_particle": {"neutron": [1, 0]}}, "invalid"),
        ({"max_particles_per_process": 0}, "max_particles"),
        ({"coupling_interface": "outer_and_inner"}, "coupling_interface"),
        ({"openmc_normal_sign_by_surface": {1: 0}}, "normal_sign"),
    ],
)
def test_invalid_or_ambiguous_instrumentation_fails_closed(updates, match):
    values = {
        "surface_ids": [1],
        "energy_edges_by_particle": {"neutron": [0, 20.0e6]},
        "openmc_normal_sign_by_surface": {1: 1},
        "max_particles_per_process": 10,
        "max_source_files": 1,
        "mpi_ranks": 1,
    }
    values.update(updates)
    with pytest.raises(ValueError, match=match):
        build_surface_instrumentation_spec(**values)
