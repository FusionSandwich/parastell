import numpy as np
import pytest

from parastell.magnet_boundary_envelope import IndependentClosure
from parastell.magnet_boundary_envelope import validate_energy_axes
from parastell.magnet_energy_architecture import condense_adjacent_groups
from parastell.magnet_energy_architecture import photon_master_edges
from parastell.multigroup_sn import LayerTransport
from parastell.multigroup_sn import solve_multilayer_sn
from parastell.openmc16 import PDG_PARTICLES
from parastell.openmc16 import add_envelope_tallies
from parastell.openmc16 import capability_report


def test_pdg_particles_are_not_legacy_enums():
    assert PDG_PARTICLES == {
        "neutron": 2112,
        "photon": 22,
        "electron": 11,
        "positron": -11,
    }


def test_openmc_016_capability_gate_and_tallies():
    pytest.importorskip("openmc")
    report = capability_report()
    if report["version"] < "0.16.0":
        pytest.skip("requires OpenMC 0.16")
    import openmc

    model = openmc.Model()
    inventory = add_envelope_tallies(
        model,
        surface_ids=[1, 2],
        cell_ids=[10],
        neutron_edges_eV=[0.0, 1.0e6, 20.0e6],
        photon_edges_eV=[1.0e3, 1.0e6, 30.0e6],
    )
    assert report["passes"]
    assert len(inventory.current) == 2
    assert len(inventory.directional_current) == 2
    assert len(inventory.surface_flux) == 2
    assert len(inventory.volume_flux) == 2
    assert len(inventory.production) == 4
    assert len(model.tallies) == 15


def test_boundary_bank_projection_and_absorption_balance():
    from types import SimpleNamespace

    import numpy as np

    from parastell.multigroup_sn import LayerTransport
    from parastell.multigroup_sn import project_incoming_bank_to_ordinates
    from parastell.multigroup_sn import solve_multilayer_sn

    bank = SimpleNamespace(
        columns={
            "particle": np.asarray(["neutron", "neutron"]),
            "mu": np.asarray([-1.0, -0.25]),
            "energy_eV": np.asarray([0.5, 1.5]),
            "weight": np.asarray([0.25, 0.75]),
        }
    )
    edges = np.asarray([0.0, 1.0, 2.0])
    angular = project_incoming_bank_to_ordinates(
        bank, "neutron", edges, ordinates=8
    )
    mu, weights = np.polynomial.legendre.leggauss(8)
    assert np.sum(angular * (mu[mu > 0] * weights[mu > 0])) == pytest.approx(
        1.0
    )
    absorption = np.asarray([0.2, 0.4])
    layer = LayerTransport(
        "absorber",
        "test",
        0.1,
        total_xs_cm_1={"neutron": absorption},
        absorption_xs_cm_1={"neutron": absorption},
        scattering_xs_cm_1={"neutron": np.zeros((2, 2))},
    )
    result = solve_multilayer_sn(
        [layer],
        {"neutron": angular},
        {"neutron": edges},
        ordinates=8,
    )
    assert result.particle_balance_error < 1.0e-12


def test_particle_energy_axes_are_separate():
    axes = validate_energy_axes(
        {
            "neutron_energy_edges_eV": [0.0, 1.0, 2.0],
            "photon_energy_edges_eV": [1.0, 10.0],
        }
    )
    assert len(axes["neutron_energy_edges_eV"]) == 3
    with pytest.raises(ValueError, match="missing"):
        validate_energy_axes({"neutron_energy_edges_eV": [0.0, 1.0]})


def test_photon_grid_is_not_ccfe_neutron_grid():
    edges = photon_master_edges(background_groups=42)
    assert len(edges) == 43
    assert edges[0] > 0.0


def test_independent_closure_never_conditions_bank():
    closure = IndependentClosure(1.0, 0.1, 1.05, 0.1, 1.04, 0.1, 1.03, 0.1)
    assert closure.passes
    assert closure.comparison("bank")["difference"] == pytest.approx(0.05)


def test_condensation_preserves_integrated_responses():
    response = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    result = condense_adjacent_groups(
        np.arange(5.0), response, maximum_relative_error=1.0e-12
    )
    assert result.source_normalization_error == 0.0
    assert result.maximum_response_error == pytest.approx(0.0)


def test_multigroup_vacuum_transmission():
    zero = np.zeros((1, 1))
    layer = LayerTransport(
        "vacuum",
        "vacuum",
        1.0,
        {"neutron": np.zeros(1)},
        {"neutron": np.zeros(1)},
        {"neutron": zero},
    )
    incoming = {"neutron": np.ones((1, 4))}
    result = solve_multilayer_sn(
        [layer], incoming, {"neutron": [0.0, 1.0]}, ordinates=8
    )
    assert result.outgoing_current == pytest.approx(result.incoming_current)
    assert result.particle_balance_error == pytest.approx(0.0)
